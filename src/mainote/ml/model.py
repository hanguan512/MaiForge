"""Audio encoder + event decoder for P3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ChartModelConfig:
    hidden: int = 256
    audio_layers: int = 4
    decoder_layers: int = 6
    heads: int = 8
    ffn: int = 1024
    dropout: float = 0.1
    max_audio_frames: int = 4096
    max_tokens: int = 2048
    n_mels: int = 80
    # Architecture v2.  v1 checkpoints (small embedding added directly to a
    # sinusoidal PE, no final LayerNorm, stride-4 audio frontend) cannot be
    # loaded into this model; retrain instead.
    embed_scale: bool = True
    final_norm: bool = True
    # Total temporal downsampling of the mel frontend.  2 keeps ~43 fps,
    # which preserves much more timing detail than the v1 stride of 4.
    audio_stride: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ChartModelConfig":
        data = data or {}
        return cls(
            hidden=int(data.get("hidden", 256)),
            audio_layers=int(data.get("audio_layers", 4)),
            decoder_layers=int(data.get("decoder_layers", 6)),
            heads=int(data.get("heads", 8)),
            ffn=int(data.get("ffn", 1024)),
            dropout=float(data.get("dropout", 0.1)),
            max_audio_frames=int(data.get("max_audio_frames", 4096)),
            max_tokens=int(data.get("max_tokens", 2048)),
            n_mels=int(data.get("n_mels", 80)),
            embed_scale=bool(data.get("embed_scale", True)),
            final_norm=bool(data.get("final_norm", True)),
            audio_stride=int(data.get("audio_stride", 2)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden,
            "audio_layers": self.audio_layers,
            "decoder_layers": self.decoder_layers,
            "heads": self.heads,
            "ffn": self.ffn,
            "dropout": self.dropout,
            "max_audio_frames": self.max_audio_frames,
            "max_tokens": self.max_tokens,
            "n_mels": self.n_mels,
            "embed_scale": self.embed_scale,
            "final_norm": self.final_norm,
            "audio_stride": self.audio_stride,
        }

    def validate(self) -> None:
        if self.audio_stride < 1 or self.audio_stride & (self.audio_stride - 1):
            raise ValueError(
                f"audio_stride must be a power of two >= 1, got {self.audio_stride}"
            )


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        if max_len < 1:
            raise ValueError("max_len must be positive")
        self.max_len = max_len
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(1, max_len, dim, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class AudioEncoder(nn.Module):
    """Mel frames -> subsampled Transformer memory."""

    def __init__(self, config: ChartModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        stride = 1
        layers: list[nn.Module] = []
        channels = config.n_mels
        while stride < config.audio_stride:
            layers.extend(
                [
                    nn.Conv1d(
                        channels,
                        config.hidden,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                    ),
                    nn.GELU(),
                ]
            )
            channels = config.hidden
            stride *= 2
        if not layers:
            # audio_stride == 1 still needs the mel -> hidden projection.
            layers.extend(
                [
                    nn.Conv1d(
                        channels,
                        config.hidden,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                    ),
                    nn.GELU(),
                ]
            )
        self.frontend = nn.Sequential(*layers)
        self.frontend_stride = config.audio_stride
        max_memory = (config.max_audio_frames + self.frontend_stride - 1) // (
            self.frontend_stride
        )
        self.pos = PositionalEncoding(
            config.hidden, max(1, max_memory + 1), config.dropout
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden,
            nhead=config.heads,
            dim_feedforward=config.ffn,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.audio_layers,
            enable_nested_tensor=False,
            norm=nn.LayerNorm(config.hidden) if config.final_norm else None,
        )

    def forward(
        self,
        mel: torch.Tensor,
        mel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # mel: (B, T, M), mel_mask: (B, T), True = padding
        x = mel.transpose(1, 2)
        x = self.frontend(x)
        x = x.transpose(1, 2)
        if x.size(1) > self.pos.max_len:
            x = x[:, : self.pos.max_len]
        x = self.pos(x)
        pooled = F.max_pool1d(
            mel_mask.float().unsqueeze(1),
            kernel_size=self.frontend_stride,
            stride=self.frontend_stride,
            ceil_mode=True,
        ).squeeze(1)
        if pooled.size(1) < x.size(1):
            pooled = F.pad(
                pooled,
                (0, x.size(1) - pooled.size(1)),
                value=1.0,
            )
        pooled = pooled[:, : x.size(1)].bool()
        if pooled.size(1) == 0:
            pooled = torch.zeros(
                (x.size(0), x.size(1)), dtype=torch.bool, device=x.device
            )
        # Never let a sample be entirely masked out.
        pooled[:, 0] = False
        x = self.encoder(x, src_key_padding_mask=pooled)
        return x, pooled


class ChartModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        config: ChartModelConfig,
        *,
        pad_id: int = 0,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.pad_id = int(pad_id)
        config.validate()
        self.audio = AudioEncoder(config)
        self.token_embed = nn.Embedding(
            vocab_size, config.hidden, padding_idx=self.pad_id
        )
        # The embedding is tied to the output projection.  PyTorch's default
        # N(0, 1) init makes initial logits ~sqrt(hidden) times too large
        # (cross-entropy starts around 250 instead of ~ln(vocab)).  Use the
        # standard small init so training starts from a near-uniform model.
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embed.weight[self.pad_id].zero_()
        # Scale embeddings to unit-ish variance before adding sinusoidal PE.
        # Without this, PE (std ~0.707) is ~25x larger than token content
        # (std ~0.028), so position dominates the residual stream.
        self.embed_scale = math.sqrt(config.hidden) if config.embed_scale else 1.0
        self.token_pos = PositionalEncoding(config.hidden, config.max_tokens, config.dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden,
            nhead=config.heads,
            dim_feedforward=config.ffn,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.decoder_layers)
        self.decoder_norm = (
            nn.LayerNorm(config.hidden) if config.final_norm else nn.Identity()
        )
        self.lm_head = nn.Linear(config.hidden, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def token_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token ids -> decoder input embeddings (shared by train and decode)."""
        return self.token_pos(self.token_embed(tokens) * self.embed_scale)

    def decode_tokens(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the decoder stack; callers only need to add ``lm_head``."""
        tgt = self.token_embeddings(tokens)
        out = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_mask,
        )
        return self.decoder_norm(out)

    def forward(
        self,
        mel: torch.Tensor,
        mel_mask: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory, memory_mask = self.audio(mel, mel_mask)
        length = tokens.size(1)
        causal = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        out = self.decode_tokens(
            tokens,
            memory,
            memory_mask=memory_mask,
            tgt_mask=causal,
            tgt_key_padding_mask=token_mask,
        )
        return self.lm_head(out)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    vocab_size: int,
    config: ChartModelConfig,
    *,
    pad_id: int = 0,
) -> ChartModel:
    return ChartModel(vocab_size=vocab_size, config=config, pad_id=pad_id)
