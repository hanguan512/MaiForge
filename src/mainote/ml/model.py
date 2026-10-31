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
        }


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int, dropout: float = 0.0):
        super().__init__()
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
        self.config = config
        self.frontend = nn.Sequential(
            nn.Conv1d(config.n_mels, config.hidden, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(config.hidden, config.hidden, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(config.hidden, config.max_audio_frames, config.dropout)
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
        if x.size(1) > self.config.max_audio_frames:
            x = x[:, : self.config.max_audio_frames]
        x = self.pos(x)
        pooled = F.max_pool1d(
            mel_mask.float().unsqueeze(1), kernel_size=4, stride=4
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
        self.lm_head = nn.Linear(config.hidden, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def forward(
        self,
        mel: torch.Tensor,
        mel_mask: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory, memory_mask = self.audio(mel, mel_mask)
        tgt = self.token_pos(self.token_embed(tokens))
        length = tokens.size(1)
        causal = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        out = self.decoder(
            tgt,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=token_mask,
            memory_key_padding_mask=memory_mask,
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
