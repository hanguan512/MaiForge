"""P3 training loop.

Run on a GPU server; this workspace has no GPU and intentionally does not train
here.  See ``docs/TRAINING.md`` for the full workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..eval.metrics import evaluate_chart
from ..maidata.model import TICKS_PER_BAR, Chart
from .dataset import ChartWindowDataset, collate_windows
from .decode import constrained_decode
from .model import ChartModelConfig, build_model
from .tokenizer import ChartVocabulary, EventTokenizer, token_category


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required for training configs. Install with: "
            "pip install pyyaml"
        ) from exc
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("training config must be a YAML mapping")
    return data


def resolve_device(value: str | None) -> torch.device:
    if value and value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(
    dataset: ChartWindowDataset,
    *,
    batch_size: int,
    num_workers: int,
    pad_id: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        collate_fn=partial(collate_windows, pad_id=pad_id),
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_token_weights(
    vocab: ChartVocabulary,
    weights: dict[str, Any] | None,
) -> torch.Tensor | None:
    """Per-vocabulary-token loss weights.

    Keys may be ``token_category`` names (``type``, ``duration``, ...) or exact
    token strings such as ``SLIDE``/``TOUCH``.  Exact tokens win over their
    category, which lets configs upweight slide and touch without changing the
    reported per-category accuracy.
    """
    if not weights:
        return None
    tensor = torch.ones(len(vocab.tokens), dtype=torch.float32)
    seen: set[str] = set()
    for token_id, token in enumerate(vocab.tokens):
        key = token if token in weights else token_category(token)
        if key not in weights:
            continue
        value = float(weights[key])
        if value <= 0:
            raise ValueError(f"token weight for {key!r} must be positive")
        tensor[token_id] = value
        seen.add(key)
    unknown = sorted(set(weights) - seen)
    if unknown:
        raise ValueError(
            "unknown token_class_weights keys: " + ", ".join(unknown)
        )
    return tensor


def sequence_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_id: int,
    *,
    token_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Window-normalized cross-entropy with optional per-token weights.

    Each window contributes equally regardless of how many tokens it has, so
    dense master charts do not dominate sparse basic charts.  Returns
    ``(loss, per_token_loss, valid_mask)``; the latter two are unweighted and
    are used for reporting.
    """
    vocab_size = logits.size(-1)
    token_loss = F.cross_entropy(
        logits.float().reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
        label_smoothing=label_smoothing,
    ).view_as(labels)
    valid = labels.ne(pad_id)
    if token_weights is not None:
        weights = token_weights.to(labels.device)[labels] * valid
    else:
        weights = valid.to(token_loss.dtype)
    denom = weights.sum(dim=1)
    per_window = (token_loss * weights).sum(dim=1) / denom.clamp_min(1.0)
    used = denom > 0
    if bool(used.any()):
        loss = per_window[used].mean()
    else:
        loss = token_loss.sum() * 0.0
    return loss, token_loss, valid


class ClassAccuracy:
    """Accumulate per-token-category accuracy for validation reporting."""

    def __init__(self, vocab: ChartVocabulary):
        self.names = sorted({token_category(t) for t in vocab.tokens})
        positions = {name: i for i, name in enumerate(self.names)}
        self.index = torch.tensor(
            [positions[token_category(t)] for t in vocab.tokens],
            dtype=torch.long,
        )
        self.correct = [0] * len(self.names)
        self.total = [0] * len(self.names)

    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        if not bool(valid.any()):
            return
        idx = self.index.to(labels.device)[labels[valid]]
        hits = preds[valid].eq(labels[valid])
        correct = torch.bincount(
            idx, weights=hits.float(), minlength=len(self.names)
        )
        total = torch.bincount(idx, minlength=len(self.names))
        self.correct = [a + int(b) for a, b in zip(self.correct, correct.cpu())]
        self.total = [a + int(b) for a, b in zip(self.total, total.cpu())]

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "correct": self.correct[i],
                "total": self.total[i],
                "accuracy": round(self.correct[i] / self.total[i], 4)
                if self.total[i]
                else None,
            }
            for i, name in enumerate(self.names)
        }


def train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(args.device or cfg.get("device"))
    output_dir = Path(args.output_dir or cfg.get("output_dir", "runs/p3_base"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    if not data_cfg.get("manifest") or not data_cfg.get("features_dir"):
        raise ValueError("config.data.manifest and config.data.features_dir are required")
    model_cfg = ChartModelConfig.from_dict(cfg.get("model"))
    model_cfg.validate()
    tokenizer = EventTokenizer()
    vocab = tokenizer.vocab
    data_max_frames = int(
        data_cfg.get("max_audio_frames", model_cfg.max_audio_frames)
    )
    if data_max_frames > model_cfg.max_audio_frames:
        raise ValueError(
            f"data.max_audio_frames={data_max_frames} exceeds "
            f"model.max_audio_frames={model_cfg.max_audio_frames}; the encoder "
            "would silently truncate the audio window"
        )
    cache_kwargs = {
        "chart_cache_size": int(data_cfg.get("chart_cache_size", 512)),
        "feature_cache_size": int(data_cfg.get("feature_cache_size", 64)),
    }

    train_dataset = ChartWindowDataset(
        data_cfg["manifest"],
        data_cfg["features_dir"],
        "train",
        tokenizer,
        window_bars=int(data_cfg.get("window_bars", 16)),
        stride_bars=int(data_cfg.get("stride_bars", 8)),
        context_bars=int(data_cfg.get("context_bars", 2)),
        min_events=int(data_cfg.get("min_events", 1)),
        max_windows_per_chart=data_cfg.get("max_windows_per_chart"),
        max_audio_frames=data_max_frames,
        min_alignment=float(data_cfg.get("min_alignment", 0.0)),
        augment=True,
        spec_augment=data_cfg.get("spec_augment"),
        **cache_kwargs,
    )
    val_dataset = ChartWindowDataset(
        data_cfg["manifest"],
        data_cfg["features_dir"],
        "val",
        tokenizer,
        window_bars=int(data_cfg.get("window_bars", 16)),
        stride_bars=int(data_cfg.get("stride_bars", 8)),
        context_bars=int(data_cfg.get("context_bars", 2)),
        min_events=int(data_cfg.get("min_events", 1)),
        max_windows_per_chart=data_cfg.get("max_windows_per_chart"),
        max_audio_frames=data_max_frames,
        min_alignment=float(data_cfg.get("min_alignment", 0.0)),
        augment=False,
        **cache_kwargs,
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            "training dataset is empty. Check manifest split/valid/audio/features."
        )

    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 4))
    train_loader = build_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pad_id=vocab.pad_id,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        build_loader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pad_id=vocab.pad_id,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        if len(val_dataset) > 0
        else None
    )

    model = build_model(
        len(vocab.tokens), model_cfg, pad_id=vocab.pad_id
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    grad_accum = max(1, int(train_cfg.get("grad_accum", 1)))
    max_steps = int(args.max_steps or train_cfg.get("max_steps") or 0)
    epochs = int(train_cfg.get("epochs", 30))
    # With --limit-batches, the final accumulated batch must still trigger an
    # optimizer step instead of being discarded.
    batches_per_epoch = len(train_loader)
    if args.limit_batches:
        batches_per_epoch = min(batches_per_epoch, args.limit_batches)
    steps_per_epoch = max(1, math.ceil(batches_per_epoch / grad_accum))
    total_steps = max_steps if max_steps else steps_per_epoch * epochs
    warmup = int(train_cfg.get("warmup_steps", 500))
    warmup_clamped_from = None
    if total_steps > 0 and warmup >= total_steps:
        warmup_clamped_from = warmup
        warmup = max(1, total_steps // 10) if total_steps > 1 else 0

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
    else:  # pragma: no cover - old PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    log_every = int(args.log_every or train_cfg.get("log_every", 20))
    save_every = int(train_cfg.get("save_every", 5))
    patience = int(train_cfg.get("patience", 8))
    token_weights = build_token_weights(
        vocab, train_cfg.get("token_class_weights")
    )
    if token_weights is not None:
        token_weights = token_weights.to(device)
    note_eval_cfg = dict(train_cfg.get("note_eval") or {})
    note_eval_enabled = bool(note_eval_cfg.get("enabled", False)) and (
        val_loader is not None
    )
    checkpoint_metric = str(
        train_cfg.get(
            "checkpoint_metric",
            "note_score" if note_eval_enabled else "val_loss",
        )
    )
    note_metrics = {
        "note_score",
        "onset_f1",
        "onset_precision",
        "onset_recall",
        "position_accuracy",
        "type_accuracy",
        "exact_accuracy",
        "slide_endpoint_accuracy",
    }
    if checkpoint_metric != "val_loss" and checkpoint_metric not in note_metrics:
        raise ValueError(
            f"unknown checkpoint_metric {checkpoint_metric!r}; use val_loss or "
            + ", ".join(sorted(note_metrics))
        )
    if checkpoint_metric in note_metrics and not note_eval_enabled:
        raise ValueError(
            f"checkpoint_metric={checkpoint_metric!r} requires "
            "train.note_eval.enabled=true and a non-empty val split"
        )
    higher_is_better = checkpoint_metric != "val_loss"

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    best_metric = float("-inf") if higher_is_better else float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("vocab_tokens") and checkpoint["vocab_tokens"] != vocab.tokens:
            raise RuntimeError("checkpoint vocabulary differs from current tokenizer")
        saved_model_cfg = checkpoint.get("model_config")
        if saved_model_cfg and saved_model_cfg != model_cfg.to_dict():
            raise RuntimeError(
                "model config differs from checkpoint. Architecture v2 "
                "(embed_scale/final_norm/audio_stride) is not backward "
                "compatible with v1 checkpoints; retrain or use the matching "
                "config."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val = float(checkpoint.get("best_val_loss", best_val))
        if checkpoint.get("best_metric_name") == checkpoint_metric:
            best_metric = float(checkpoint.get("best_metric", best_metric))

    print(
        json.dumps(
            {
                "device": str(device),
                "train_windows": len(train_dataset),
                "val_windows": len(val_dataset),
                "params": model.parameter_count(),
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "total_steps": total_steps,
                "warmup": warmup,
                "warmup_clamped_from": warmup_clamped_from,
                "amp": amp,
                "audio_stride": model_cfg.audio_stride,
                "embed_scale": model_cfg.embed_scale,
                "final_norm": model_cfg.final_norm,
                "token_class_weights": train_cfg.get("token_class_weights"),
                "note_eval": note_eval_enabled,
                "checkpoint_metric": checkpoint_metric,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.dry_run:
        batch = next(iter(train_loader))
        batch = move_batch(batch, device)
        tokens = batch["tokens"]
        logits = model(
            batch["mel"],
            batch["mel_mask"],
            tokens[:, :-1],
            batch["token_mask"][:, :-1],
        )
        loss, _, _ = sequence_cross_entropy(
            logits,
            tokens[:, 1:],
            vocab.pad_id,
            token_weights=token_weights,
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "mel": list(batch["mel"].shape),
                    "tokens": list(tokens.shape),
                    "logits": list(logits.shape),
                    "loss": float(loss.item()),
                },
                ensure_ascii=False,
            )
        )
        return 0

    metrics_path = output_dir / "metrics.jsonl"
    metrics_fh = metrics_path.open("a", encoding="utf-8")
    stop = False
    epochs_without_improvement = 0
    for epoch in range(start_epoch, epochs):
        if stop:
            break
        model.train()
        train_loss = 0.0
        train_loss_windows = 0
        train_token_loss = 0.0
        train_correct = 0
        train_tokens = 0
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            if args.limit_batches and batch_index >= args.limit_batches:
                break
            batch = move_batch(batch, device)
            tokens = batch["tokens"]
            if tokens.size(1) < 2:
                continue
            input_ids = tokens[:, :-1]
            labels = tokens[:, 1:]
            token_mask = batch["token_mask"][:, :-1]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                logits = model(
                    batch["mel"], batch["mel_mask"], input_ids, token_mask
                )
                loss, token_loss, valid = sequence_cross_entropy(
                    logits,
                    labels,
                    vocab.pad_id,
                    token_weights=token_weights,
                    label_smoothing=label_smoothing,
                )
            scaler.scale(loss / grad_accum).backward()
            preds = logits.argmax(dim=-1)
            n_windows = int(valid.any(dim=1).sum().item())
            train_correct += int((preds.eq(labels) & valid).sum().item())
            train_tokens += int(valid.sum().item())
            train_loss += float(loss.item()) * n_windows
            train_loss_windows += n_windows
            train_token_loss += float(token_loss[valid].sum().item())
            is_step = (batch_index + 1) % grad_accum == 0 or (
                batch_index + 1
            ) == batches_per_epoch
            if is_step:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if global_step % log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step": global_step,
                                "train_loss": round(
                                    train_loss / max(1, train_loss_windows), 4
                                ),
                                "train_token_ce": round(
                                    train_token_loss / max(1, train_tokens), 4
                                ),
                                "train_acc": round(
                                    train_correct / max(1, train_tokens), 4
                                ),
                                "lr": round(lr, 8),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if max_steps and global_step >= max_steps:
                    stop = True
                    break
        train_loss = train_loss / max(1, train_loss_windows)
        train_token_ce = train_token_loss / max(1, train_tokens)
        train_acc = train_correct / max(1, train_tokens)

        val_loss = None
        val_acc = None
        val_class_accuracy = None
        if val_loader is not None:
            val_loss, val_acc, val_class_accuracy = evaluate(
                model,
                val_loader,
                device,
                pad_id=vocab.pad_id,
                vocab=vocab,
                amp=amp,
                token_weights=token_weights,
                limit_batches=args.limit_batches,
            )
            if val_loss is not None:
                best_val = min(best_val, val_loss)

        note_metrics = None
        if note_eval_enabled:
            every = max(1, int(note_eval_cfg.get("every_n_epochs", 5)))
            if epoch % every == 0 or epoch + 1 >= epochs:
                note_metrics = evaluate_note_level(
                    model,
                    val_dataset,
                    tokenizer,
                    device,
                    max_songs=int(note_eval_cfg.get("max_songs", 2)),
                    max_windows=int(note_eval_cfg.get("max_windows", 4)),
                    max_new_tokens=int(
                        note_eval_cfg.get("max_new_tokens", 2048)
                    ),
                    temperature=float(note_eval_cfg.get("temperature", 0.0)),
                    top_k=note_eval_cfg.get("top_k"),
                    tolerance_seconds=float(
                        note_eval_cfg.get("tolerance_seconds", 0.06)
                    ),
                    seed=int(note_eval_cfg.get("seed", 0)),
                )

        elapsed = time.time() - epoch_start
        record = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": round(train_loss, 4),
            "train_token_ce": round(train_token_ce, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "val_acc": round(val_acc, 4) if val_acc is not None else None,
            "val_class_accuracy": val_class_accuracy,
            "note_eval": note_metrics,
            "seconds": round(elapsed, 2),
        }
        print(json.dumps(record, ensure_ascii=False), flush=True)
        metrics_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        metrics_fh.flush()

        monitor: float | None
        if checkpoint_metric == "val_loss":
            monitor = val_loss if val_loss is not None else train_loss
        else:
            monitor = (
                float(note_metrics[checkpoint_metric])
                if note_metrics
                and note_metrics.get(checkpoint_metric) is not None
                else None
            )
        metric_available = monitor is not None
        improved = False
        if metric_available:
            if higher_is_better:
                improved = monitor > best_metric
            else:
                improved = monitor < best_metric
            if improved:
                best_metric = monitor
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            best_val,
            best_metric,
            checkpoint_metric,
            cfg,
            vocab,
            model_cfg,
        )
        if improved:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_val,
                best_metric,
                checkpoint_metric,
                cfg,
                vocab,
                model_cfg,
            )
        if save_every and (epoch + 1) % save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch + 1:03d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_val,
                best_metric,
                checkpoint_metric,
                cfg,
                vocab,
                model_cfg,
            )
        if patience and epochs_without_improvement >= patience:
            print(
                json.dumps(
                    {
                        "early_stop": True,
                        "epoch": epoch,
                        "best_val_loss": round(best_val, 4),
                        "best_metric": checkpoint_metric,
                        "best_metric_value": _round_metric(best_metric),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break

    metrics_fh.close()
    print(
        json.dumps(
            {
                "done": True,
                "best_val_loss": round(best_val, 4),
                "best_metric": checkpoint_metric,
                "best_metric_value": _round_metric(best_metric),
                "steps": global_step,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _round_metric(value: float) -> float | None:
    if math.isinf(value):
        return None
    return round(value, 4)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    *,
    pad_id: int,
    vocab: ChartVocabulary,
    amp: bool,
    token_weights: torch.Tensor | None = None,
    limit_batches: int | None = None,
) -> tuple[float | None, float | None, dict[str, dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    loss_windows = 0
    total_tokens = 0
    correct = 0
    class_accuracy = ClassAccuracy(vocab)
    for batch_index, batch in enumerate(loader):
        if limit_batches and batch_index >= limit_batches:
            break
        batch = move_batch(batch, device)
        tokens = batch["tokens"]
        if tokens.size(1) < 2:
            continue
        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            logits = model(
                batch["mel"], batch["mel_mask"], input_ids, batch["token_mask"][:, :-1]
            )
            loss, _, valid = sequence_cross_entropy(
                logits,
                labels,
                pad_id,
                token_weights=token_weights,
            )
        windows = int(valid.any(dim=1).sum().item())
        total_loss += float(loss.item()) * windows
        loss_windows += windows
        total_tokens += int(valid.sum().item())
        preds = logits.argmax(dim=-1)
        correct += int((preds.eq(labels) & valid).sum().item())
        class_accuracy.update(preds, labels, valid)
    if loss_windows == 0:
        return None, None, class_accuracy.to_dict()
    return (
        total_loss / loss_windows,
        correct / max(1, total_tokens),
        class_accuracy.to_dict(),
    )


@torch.no_grad()
def evaluate_note_level(
    model,
    dataset: ChartWindowDataset,
    tokenizer: EventTokenizer,
    device: torch.device,
    *,
    max_songs: int = 2,
    max_windows: int = 4,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_k: int | None = None,
    tolerance_seconds: float = 0.06,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Decode a few validation windows and compare notes to the reference.

    Token accuracy is not chart quality; this is the metric used to pick
    ``best.pt`` when enabled.
    """
    if len(dataset) == 0:
        return None
    max_songs = max(1, int(max_songs))
    max_windows = max(1, int(max_windows))
    model.eval()
    vocab = tokenizer.vocab
    groups: dict[str, list[int]] = {}
    for index, spec in enumerate(dataset.windows):
        groups.setdefault(str(spec.row.get("song_id", "")), []).append(index)
    chosen: list[int] = []
    for song_id in list(groups)[: max(1, max_songs)]:
        indices = groups[song_id]
        if len(indices) > max_windows:
            step = max(1, len(indices) // max_windows)
            indices = indices[::step][:max_windows]
        chosen.extend(indices)

    results: list[dict[str, Any]] = []
    decode_errors = 0
    for index in chosen:
        item = dataset[index]
        token_strings = vocab.decode(item["tokens"])
        difficulty_index = 5
        level: float | None = None
        for token in token_strings[:4]:
            if token.startswith("DIFF_"):
                difficulty_index = int(token[len("DIFF_") :])
            elif token.startswith("LEVEL_"):
                level = int(token[len("LEVEL_") :]) / 2.0
        mel = (
            torch.from_numpy(item["mel"])
            .float()
            .unsqueeze(0)
            .to(device)
        )
        mel_mask = torch.zeros(
            (1, mel.size(1)), dtype=torch.bool, device=device
        )
        try:
            _, parsed = constrained_decode(
                model,
                mel,
                mel_mask,
                tokenizer=tokenizer,
                difficulty_index=difficulty_index,
                level=level,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
                allow_incomplete=True,
            )
        except Exception:  # noqa: BLE001 - keep evaluation running
            decode_errors += 1
            continue
        reference = tokenizer.decode_ids(item["tokens"])
        template = dataset.chart_for(index)
        start_tick = dataset.window_bounds(index)[0]
        for event in reference.events:
            event.tick += start_tick
        for event in parsed.events:
            event.tick += start_tick
        ref_chart = _window_chart(template, reference.events)
        pred_chart = _window_chart(template, parsed.events)
        results.append(
            evaluate_chart(
                pred_chart,
                ref_chart,
                tolerance_seconds=tolerance_seconds,
            )
        )
    return _aggregate_note_results(results, decode_errors)


def _window_chart(template: Chart, events: list[Any]) -> Chart:
    return Chart(
        index=template.index,
        level=template.level,
        designer=template.designer,
        first=template.first,
        initial_bpm=template.initial_bpm,
        events=events,
        bpm_changes=list(template.bpm_changes),
        division_changes=list(template.division_changes),
        end_tick=template.end_tick,
    )


def _aggregate_note_results(
    results: list[dict[str, Any]],
    decode_errors: int,
) -> dict[str, Any] | None:
    if not results:
        return None
    matched = sum(r["counts"]["matched"] for r in results)
    reference = sum(r["counts"]["reference"] for r in results)
    predicted = sum(r["counts"]["predicted"] for r in results)
    precision = matched / predicted if predicted else 0.0
    recall = matched / reference if reference else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    position = (
        sum(r["position_accuracy"] * r["counts"]["matched"] for r in results)
        / matched
        if matched
        else 0.0
    )
    type_accuracy = (
        sum(r["type_accuracy"] * r["counts"]["matched"] for r in results)
        / matched
        if matched
        else 0.0
    )
    exact = (
        sum(r["exact_accuracy"] * r["counts"]["matched"] for r in results)
        / matched
        if matched
        else 0.0
    )
    time_error = (
        sum(
            r["onset"]["mean_time_error_ms"] * r["counts"]["matched"]
            for r in results
        )
        / matched
        if matched
        else 0.0
    )
    slide_matched = sum(
        (r["slide"]["matched"] or 0) for r in results
    )
    slide_endpoint = (
        sum(
            (r["slide"]["endpoint_accuracy"] or 0.0)
            * (r["slide"]["matched"] or 0)
            for r in results
        )
        / slide_matched
        if slide_matched
        else None
    )
    components = [f1, position, type_accuracy]
    if slide_endpoint is not None:
        components.append(slide_endpoint)
    note_score = sum(components) / len(components)
    chord_excess_ticks = sum(
        r["chord"]["predicted_excess_ticks"] for r in results
    )
    chord_outer_ticks = sum(r["chord"]["predicted_outer_ticks"] for r in results)
    return {
        "windows": len(results),
        "decode_errors": decode_errors,
        "counts": {
            "reference": reference,
            "predicted": predicted,
            "matched": matched,
        },
        "onset_f1": round(f1, 4),
        "onset_precision": round(precision, 4),
        "onset_recall": round(recall, 4),
        "position_accuracy": round(position, 4),
        "type_accuracy": round(type_accuracy, 4),
        "exact_accuracy": round(exact, 4),
        "mean_time_error_ms": round(time_error, 3),
        "slide_endpoint_accuracy": round(slide_endpoint, 4)
        if slide_endpoint is not None
        else None,
        # The strict decoder caps outer-button chords, so this should be zero;
        # a non-zero value means a decode/repair regression slipped through.
        "max_outer_simultaneous": max(
            (r["chord"]["predicted_max_outer"] for r in results), default=0
        ),
        "chord_excess_ticks": chord_excess_ticks,
        "chord_excess_rate": round(
            chord_excess_ticks / chord_outer_ticks, 4
        )
        if chord_outer_ticks
        else 0.0,
        "note_score": round(note_score, 4),
    }


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    best_metric: float,
    best_metric_name: str,
    cfg: dict[str, Any],
    vocab: ChartVocabulary,
    model_cfg: ChartModelConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "best_metric": best_metric,
            "best_metric_name": best_metric_name,
            "config": cfg,
            "model_config": model_cfg.to_dict(),
            "vocab_tokens": vocab.tokens,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the P3 chart model")
    parser.add_argument("--config", default="configs/p3_base.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
