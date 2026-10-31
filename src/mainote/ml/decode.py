"""Grammar-constrained decoding for the P3 event model."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .model import ChartModel
from .tokenizer import (
    ChartTokenGrammar,
    ChartVocabulary,
    EventTokenizer,
    ParsedSequence,
    TokenizeError,
    level_bucket,
)


@torch.no_grad()
def greedy_decode(
    model: ChartModel,
    mel: torch.Tensor,
    mel_mask: torch.Tensor,
    *,
    tokenizer: EventTokenizer,
    difficulty_index: int,
    level: float | str | None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int = 0,
    allow_incomplete: bool = False,
) -> tuple[list[int], ParsedSequence]:
    """Generate one chart/window token sequence with the token grammar.

    ``mel`` is shape ``(1, T, n_mels)`` and ``mel_mask`` is ``(1, T)`` with
    ``True`` for padding.
    """
    device = next(model.parameters()).device
    vocab = tokenizer.vocab
    prompt = [
        "BOS",
        f"DIFF_{difficulty_index}",
        f"LEVEL_{level_bucket(level)}",
    ]
    grammar = ChartTokenGrammar(vocab)
    for token in prompt:
        grammar.feed(token)
    ids = vocab.encode(prompt)
    memory, memory_mask = model.audio(mel, mel_mask)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if temperature > 0:
        torch.manual_seed(seed)

    for _ in range(max_new_tokens):
        if grammar.parsed.done:
            break
        allowed = grammar.allowed_tokens()
        # The prompt already supplied these; don't let the model repeat them.
        allowed.discard("BOS")
        allowed = {
            t
            for t in allowed
            if not t.startswith("DIFF_") and not t.startswith("LEVEL_")
        }
        if not grammar.parsed.events:
            allowed.discard("EOS")
        if not allowed:
            break
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        tgt = model.token_pos(model.token_embed(input_ids))
        length = input_ids.size(1)
        causal = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=device),
            diagonal=1,
        )
        out = model.decoder(
            tgt,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=token_mask,
            memory_key_padding_mask=memory_mask,
        )
        logits = model.lm_head(out)[0, -1]
        allowed_ids = [vocab.token_to_id[t] for t in allowed if t in vocab.token_to_id]
        if not allowed_ids:
            break
        mask = torch.full_like(logits, float("-inf"))
        mask[allowed_ids] = 0.0
        logits = logits + mask
        if temperature <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            scaled = logits / max(temperature, 1e-6)
            if top_k:
                values, indices = torch.topk(scaled, min(top_k, scaled.numel()))
                probs = F.softmax(values, dim=-1)
                choice = torch.multinomial(
                    probs.cpu(), 1, generator=generator
                ).item()
                next_id = int(indices[choice].item())
            else:
                probs = F.softmax(scaled, dim=-1)
                next_id = int(
                    torch.multinomial(probs.cpu(), 1, generator=generator).item()
                )
        token = vocab.id_to_token[next_id]
        ids.append(next_id)
        grammar.feed(token)
        if token == "EOS":
            break

    if not grammar.parsed.done:
        if grammar.state not in ("time", "done"):
            if allow_incomplete:
                return ids, grammar.parsed
            raise TokenizeError(
                f"decoding stopped in incomplete grammar state {grammar.state!r}"
            )
    return ids, grammar.finish()
