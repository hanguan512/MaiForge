"""Describe the P3 chart model for the web UI.

This module deliberately avoids importing torch: the training page may be
opened on a machine that only has the ``web`` extra installed.  Layer shapes
and parameter counts are derived from the same formulas PyTorch uses for the
modules in :mod:`mainote.ml.model`.
"""

from __future__ import annotations

from typing import Any

from ..ml.tokenizer import EventTokenizer

DEFAULT_ARCHITECTURE = {
    "hidden": 256,
    "audio_layers": 4,
    "decoder_layers": 6,
    "heads": 8,
    "ffn": 1024,
    "dropout": 0.1,
    "max_audio_frames": 4096,
    "max_tokens": 2048,
    "n_mels": 80,
    "audio_stride": 2,
    "embed_scale": True,
    "final_norm": True,
}

# prepare-features defaults: sr=22050, hop=256.
MEL_FPS = 22050.0 / 256.0


def _linear_params(fan_in: int, fan_out: int) -> int:
    return fan_in * fan_out + fan_out


def _layer_norm_params(dim: int) -> int:
    return 2 * dim


def _attention_params(dim: int) -> int:
    # in_proj (Q, K, V) + out_proj, with biases.
    return 4 * dim * dim + 4 * dim


def _encoder_layer_params(dim: int, ffn: int) -> int:
    # norm_first TransformerEncoderLayer: self-attn, FFN, two LayerNorms.
    return (
        _attention_params(dim)
        + _linear_params(dim, ffn)
        + _linear_params(ffn, dim)
        + 2 * _layer_norm_params(dim)
    )


def _decoder_layer_params(dim: int, ffn: int) -> int:
    # TransformerDecoderLayer: masked self-attn, cross-attn, FFN, three norms.
    return (
        2 * _attention_params(dim)
        + _linear_params(dim, ffn)
        + _linear_params(ffn, dim)
        + 3 * _layer_norm_params(dim)
    )


def _conv_params(fan_in: int, fan_out: int, kernel: int = 5) -> int:
    return fan_in * fan_out * kernel + fan_out


def _fmt(value: int) -> str:
    return f"{value:,}"


def describe_architecture(**overrides: Any) -> dict[str, Any]:
    """Return layer groups, tensor shapes and parameter estimates.

    ``overrides`` accepts the keys of :data:`DEFAULT_ARCHITECTURE`.
    """
    config = dict(DEFAULT_ARCHITECTURE)
    for key, value in overrides.items():
        if value is not None and key in config:
            config[key] = value

    hidden = int(config["hidden"])
    audio_layers = int(config["audio_layers"])
    decoder_layers = int(config["decoder_layers"])
    heads = int(config["heads"])
    ffn = int(config["ffn"])
    n_mels = int(config["n_mels"])
    audio_stride = int(config["audio_stride"])
    max_audio_frames = int(config["max_audio_frames"])
    max_tokens = int(config["max_tokens"])
    dropout = float(config["dropout"])
    final_norm = bool(config["final_norm"])
    embed_scale = bool(config["embed_scale"])

    if hidden < 8 or hidden > 8192:
        raise ValueError("hidden 必须在 8 到 8192 之间")
    if heads < 1 or hidden % heads:
        raise ValueError("hidden 必须能被 heads 整除")
    if audio_layers < 1 or audio_layers > 128:
        raise ValueError("audio_layers 必须在 1 到 128 之间")
    if decoder_layers < 1 or decoder_layers > 128:
        raise ValueError("decoder_layers 必须在 1 到 128 之间")
    if ffn < 1 or ffn > 65536:
        raise ValueError("ffn 必须在 1 到 65536 之间")
    if n_mels < 1 or n_mels > 1024:
        raise ValueError("n_mels 必须在 1 到 1024 之间")
    if audio_stride < 1 or audio_stride > 1024 or audio_stride & (audio_stride - 1):
        raise ValueError("audio_stride 必须是 2 的幂")
    if max_audio_frames < 1 or max_tokens < 1:
        raise ValueError("max_audio_frames 和 max_tokens 必须为正数")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout 必须在 [0, 1) 内")

    vocab_size = len(EventTokenizer().vocab.tokens)
    conv_steps = 0
    stride = 1
    while stride < audio_stride:
        stride *= 2
        conv_steps += 1
    if conv_steps == 0:
        conv_steps = 1

    encoder_layer_params = _encoder_layer_params(hidden, ffn)
    decoder_layer_params = _decoder_layer_params(hidden, ffn)
    conv_params = sum(
        _conv_params(n_mels if index == 0 else hidden, hidden)
        for index in range(conv_steps)
    )
    norm_params = _layer_norm_params(hidden) if final_norm else 0
    embedding_params = vocab_size * hidden
    total_params = (
        conv_params
        + audio_layers * encoder_layer_params
        + norm_params
        + embedding_params
        + decoder_layers * decoder_layer_params
        + norm_params
    )

    stride_text = f"T/{audio_stride}" if audio_stride > 1 else "T"
    groups: list[dict[str, Any]] = []

    groups.append(
        {
            "id": "input",
            "label": "输入",
            "column": "encoder",
            "layers": [
                {
                    "id": "mel",
                    "name": "Log-mel 频谱",
                    "kind": "input",
                    "shape": f"B × T × {n_mels}",
                    "detail": f"{MEL_FPS:.1f} fps · 最长 {_fmt(max_audio_frames)} 帧",
                    "params": 0,
                }
            ],
        }
    )

    conv_layers = []
    fan_in = n_mels
    for index in range(conv_steps):
        divisor = 2 ** (index + 1)
        conv_shape = f"B × {'T' if divisor == 1 else f'T/{divisor}'} × {hidden}"
        conv_layers.append(
            {
                "id": f"conv{index + 1}",
                "name": f"Conv1d #{index + 1}",
                "kind": "conv",
                "shape": conv_shape,
                "detail": f"kernel 5 · stride 2 · GELU · {fan_in} → {hidden}",
                "params": _conv_params(fan_in, hidden),
            }
        )
        fan_in = hidden
    groups.append(
        {
            "id": "frontend",
            "label": "音频前端",
            "column": "encoder",
            "layers": conv_layers,
        }
    )

    encoder_layers = [
        {
            "id": "audio-pos",
            "name": "正弦位置编码",
            "kind": "pos",
            "shape": f"B × {stride_text} × {hidden}",
            "detail": f"dropout {dropout:g} · 无可训练参数",
            "params": 0,
        }
    ]
    for index in range(audio_layers):
        encoder_layers.append(
            {
                "id": f"enc{index + 1}",
                "name": f"Encoder Layer {index + 1}",
                "kind": "encoder",
                "shape": f"B × {stride_text} × {hidden}",
                "detail": (
                    f"Self-Attn {heads} heads · FFN {ffn} · GELU · "
                    f"LayerNorm ×2 · norm_first"
                ),
                "params": encoder_layer_params,
            }
        )
    if final_norm:
        encoder_layers.append(
            {
                "id": "enc-norm",
                "name": "Final LayerNorm",
                "kind": "norm",
                "shape": f"B × {stride_text} × {hidden}",
                "detail": "编码器 memory 输出",
                "params": _layer_norm_params(hidden),
            }
        )
    groups.append(
        {
            "id": "encoder",
            "label": "音频编码器",
            "column": "encoder",
            "layers": encoder_layers,
        }
    )

    groups.append(
        {
            "id": "decoder-input",
            "label": "解码器输入",
            "column": "decoder",
            "layers": [
                {
                    "id": "token-embed",
                    "name": "Token Embedding",
                    "kind": "embed",
                    "shape": f"B × L × {hidden}",
                    "detail": (
                        f"vocab {_fmt(vocab_size)} × {hidden}"
                        + (" · scale √hidden" if embed_scale else "")
                    ),
                    "params": embedding_params,
                },
                {
                    "id": "token-pos",
                    "name": "正弦位置编码",
                    "kind": "pos",
                    "shape": f"B × L × {hidden}",
                    "detail": f"dropout {dropout:g} · 最长 {_fmt(max_tokens)} tokens",
                    "params": 0,
                },
            ],
        }
    )

    decoder_layers_list = []
    for index in range(decoder_layers):
        decoder_layers_list.append(
            {
                "id": f"dec{index + 1}",
                "name": f"Decoder Layer {index + 1}",
                "kind": "decoder",
                "shape": f"B × L × {hidden}",
                "detail": (
                    f"Masked Self-Attn · Cross-Attn {heads} heads · "
                    f"FFN {ffn} · LayerNorm ×3 · norm_first"
                ),
                "params": decoder_layer_params,
            }
        )
    if final_norm:
        decoder_layers_list.append(
            {
                "id": "dec-norm",
                "name": "Final LayerNorm",
                "kind": "norm",
                "shape": f"B × L × {hidden}",
                "detail": "解码器输出",
                "params": _layer_norm_params(hidden),
            }
        )
    groups.append(
        {
            "id": "decoder",
            "label": "事件解码器",
            "column": "decoder",
            "layers": decoder_layers_list,
        }
    )

    groups.append(
        {
            "id": "output",
            "label": "输出",
            "column": "decoder",
            "layers": [
                {
                    "id": "lm-head",
                    "name": "LM Head（与 Embedding 共享）",
                    "kind": "head",
                    "shape": f"B × L × {_fmt(vocab_size)}",
                    "detail": "线性映射到事件词表 · 不计额外参数",
                    "params": 0,
                }
            ],
        }
    )

    return {
        "config": config,
        "vocab_size": vocab_size,
        "total_params": total_params,
        "mel_fps": MEL_FPS,
        "encoded_fps": MEL_FPS / audio_stride,
        "groups": groups,
    }
