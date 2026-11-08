"""Fixed feature vectors for structured chart tokens.

The tokenizer deliberately keeps positions and durations as opaque tokens
(``P_1`` ... ``P_8``, ``S_A1`` ..., ``DUR_0`` ... ``DUR_2048``).  That keeps
the grammar and IR simple, but it also means the model has to learn basic
geometry from scratch: ``P_1`` and ``P_2`` are unrelated embedding rows even
though they are adjacent buttons on the playfield, and ``DUR_96`` and
``DUR_97`` are unrelated classes even though they are one tick apart.

This module derives small continuous feature vectors from the token *strings*
so the model can add them to the token embedding without changing the
vocabulary.  Nothing here is learned; the tables are deterministic functions
of the vocabulary.

Position features (8 dims)::

    [sin(theta), cos(theta), button, ring_A, ring_B, ring_C, ring_D, ring_E]

Buttons 1..8 sit on the 8-button ring, so their angle is
``2*pi*(index-1)/8`` (button 1 at the top).  Numbered touch sensors use the
same sector angle inside their lettered ring.  ``C``/``C1``/``C2`` live in the
centre of the panel and therefore have no angular component; the ring one-hot
still distinguishes them.

Duration features (4 dims)::

    [log1p(units) / log1p(MAX_DUR_UNITS), units / MAX_DUR_UNITS,
     units == 0, units >= MAX_DUR_UNITS]

``DUR_NONE`` maps to all zeros; the token embedding still carries its special
meaning.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .tokenizer import MAX_DUR_UNITS

POSITION_FEATURE_DIM = 8
DURATION_FEATURE_DIM = 4

# Ring one-hot order after [sin, cos]: button, A, B, C, D, E.
_RING_ORDER = "ABCDE"
_RING_OFFSET = 3  # 0 = sin, 1 = cos, 2 = button
_SENSOR_ANGLE_RINGS = frozenset("ABDE")
_BUTTON_COUNT = 8


def _angle_features(index: int) -> tuple[float, float]:
    if not 1 <= index <= _BUTTON_COUNT:
        return 0.0, 0.0
    angle = 2.0 * math.pi * (index - 1) / _BUTTON_COUNT
    return math.sin(angle), math.cos(angle)


def _position_row(token: str) -> list[float]:
    row = [0.0] * POSITION_FEATURE_DIM
    if token.startswith("P_"):
        try:
            index = int(token[2:])
        except ValueError:
            return row
        if 1 <= index <= _BUTTON_COUNT:
            row[0], row[1] = _angle_features(index)
            row[2] = 1.0
        return row

    if not token.startswith("S_"):
        return row
    sensor = token[2:]
    if not sensor:
        return row
    ring = sensor[0]
    if ring not in _RING_ORDER:
        return row
    row[_RING_OFFSET + _RING_ORDER.index(ring)] = 1.0
    # Numbered A/B/D/E sensors follow the same sector layout as the buttons.
    # C1/C2 are centre sensors, so they intentionally keep sin=cos=0.
    if ring in _SENSOR_ANGLE_RINGS and sensor[1:].isdigit():
        row[0], row[1] = _angle_features(int(sensor[1:]))
    return row


def _duration_row(token: str) -> list[float]:
    row = [0.0] * DURATION_FEATURE_DIM
    if not token.startswith("DUR_"):
        return row
    suffix = token[len("DUR_") :]
    if suffix == "NONE":
        return row
    try:
        units = int(suffix)
    except ValueError:
        return row
    units = max(0, min(MAX_DUR_UNITS, units))
    row[0] = math.log1p(units) / math.log1p(MAX_DUR_UNITS)
    row[1] = units / MAX_DUR_UNITS
    row[2] = 1.0 if units == 0 else 0.0
    row[3] = 1.0 if units >= MAX_DUR_UNITS else 0.0
    return row


def build_position_features(tokens: Sequence[str]) -> torch.Tensor:
    """Return a ``(len(tokens), POSITION_FEATURE_DIM)`` float tensor."""
    return torch.tensor([_position_row(token) for token in tokens], dtype=torch.float32)


def build_duration_features(tokens: Sequence[str]) -> torch.Tensor:
    """Return a ``(len(tokens), DURATION_FEATURE_DIM)`` float tensor."""
    return torch.tensor([_duration_row(token) for token in tokens], dtype=torch.float32)
