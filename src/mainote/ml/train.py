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

from .dataset import ChartWindowDataset, collate_windows
from .model import ChartModelConfig, build_model
from .tokenizer import ChartVocabulary, EventTokenizer


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
    tokenizer = EventTokenizer()
    vocab = tokenizer.vocab

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
        max_audio_frames=int(data_cfg.get("max_audio_frames", model_cfg.max_audio_frames)),
        min_alignment=float(data_cfg.get("min_alignment", 0.0)),
        augment=True,
        spec_augment=data_cfg.get("spec_augment"),
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
        max_audio_frames=int(data_cfg.get("max_audio_frames", model_cfg.max_audio_frames)),
        min_alignment=float(data_cfg.get("min_alignment", 0.0)),
        augment=False,
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
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = max_steps if max_steps else steps_per_epoch * epochs
    warmup = int(train_cfg.get("warmup_steps", 500))

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

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("vocab_tokens") and checkpoint["vocab_tokens"] != vocab.tokens:
            raise RuntimeError("checkpoint vocabulary differs from current tokenizer")
        saved_model_cfg = checkpoint.get("model_config")
        if saved_model_cfg and saved_model_cfg != model_cfg.to_dict():
            raise RuntimeError(
                "model config differs from checkpoint; use the checkpoint config"
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
                "amp": amp,
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
        loss = F.cross_entropy(
            logits.reshape(-1, len(vocab.tokens)),
            tokens[:, 1:].reshape(-1),
            ignore_index=vocab.pad_id,
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
                loss = F.cross_entropy(
                    logits.reshape(-1, len(vocab.tokens)),
                    labels.reshape(-1),
                    ignore_index=vocab.pad_id,
                    label_smoothing=label_smoothing,
                )
            scaler.scale(loss / grad_accum).backward()
            valid = labels.ne(vocab.pad_id)
            preds = logits.argmax(dim=-1)
            train_correct += int((preds.eq(labels) & valid).sum().item())
            train_tokens += int(valid.sum().item())
            train_loss += float(loss.item()) * int(valid.sum().item())
            is_step = (batch_index + 1) % grad_accum == 0 or (
                batch_index + 1
            ) == len(train_loader)
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
                                    train_loss / max(1, train_tokens), 4
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
        train_loss = train_loss / max(1, train_tokens)
        train_acc = train_correct / max(1, train_tokens)

        val_loss = None
        val_acc = None
        if val_loader is not None:
            val_loss, val_acc = evaluate(
                model,
                val_loader,
                device,
                pad_id=vocab.pad_id,
                vocab_size=len(vocab.tokens),
                amp=amp,
                limit_batches=args.limit_batches,
            )
        elapsed = time.time() - epoch_start
        record = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "val_acc": round(val_acc, 4) if val_acc is not None else None,
            "seconds": round(elapsed, 2),
        }
        print(json.dumps(record, ensure_ascii=False), flush=True)
        metrics_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        metrics_fh.flush()

        monitor = val_loss if val_loss is not None else train_loss
        improved = monitor < best_val
        if improved:
            best_val = monitor
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
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break

    metrics_fh.close()
    print(
        json.dumps(
            {"done": True, "best_val_loss": round(best_val, 4), "steps": global_step},
            ensure_ascii=False,
        )
    )
    return 0


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    *,
    pad_id: int,
    vocab_size: int,
    amp: bool,
    limit_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
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
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
                ignore_index=pad_id,
            )
        valid = labels.ne(pad_id)
        count = int(valid.sum().item())
        total_loss += float(loss.item()) * count
        total_tokens += count
        preds = logits.argmax(dim=-1)
        correct += int((preds.eq(labels) & valid).sum().item())
    return (
        total_loss / max(1, total_tokens),
        correct / max(1, total_tokens),
    )


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
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
