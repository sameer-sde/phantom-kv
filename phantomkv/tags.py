"""Provenance tags as stored in a real vLLM KV cache (pure Python, no torch).

Each KV slot's first TAG_WIDTH floats hold:

    [MAGIC, pos, token, h0, h1, h2, h3, 0]

where h0..h3 are the 16-bit chunks of the 64-bit chain hash for the prefix
ending at `pos`. Every field is an integer below 2**24, so it round-trips
exactly through float32, the KV-cache dtype Phantom runs with.

The model (vllm_model.py) and the harness (reference outputs) both use this
module, so "what Phantom writes" and "what we expect" can't drift apart.
"""

from __future__ import annotations

from .provenance import ROOT, _h, chain_step

MAGIC = 4242
TAG_WIDTH = 8
_MASK16 = 0xFFFF


def encode(pos: int, token: int, chain: int) -> list[float]:
    if not (0 <= pos < 1 << 24 and 0 <= token < 1 << 24):
        raise ValueError("pos and token must fit in 24 bits to stay exact in float32")
    hs = [(chain >> (16 * i)) & _MASK16 for i in range(4)]
    return [float(MAGIC), float(pos), float(token), *map(float, hs), 0.0]


def decode(vals: list[float]) -> tuple[int, int, int] | None:
    """Return (pos, token, chain), or None if the slot holds no Phantom tag."""
    if int(vals[0]) != MAGIC:
        return None
    chain = 0
    for i in range(4):
        chain |= (int(vals[3 + i]) & _MASK16) << (16 * i)
    return int(vals[1]), int(vals[2]), chain


def next_token(chain: int, vocab_size: int) -> int:
    """Phantom's 'LM head': the token that follows the prefix `chain`."""
    return _h(chain, 0xA11CE) % vocab_size


def expected_continuation(prompt: list[int], n: int, vocab_size: int) -> list[int]:
    """What a correct engine must generate for `prompt` (greedy, n tokens)."""
    h = ROOT
    for t in prompt:
        h = chain_step(h, t)
    out = []
    for _ in range(n):
        t = next_token(h, vocab_size)
        out.append(t)
        h = chain_step(h, t)
    return out
