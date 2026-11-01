"""Provenance tags: the only "data" a Phantom KV cache ever holds.

A real model writes floating-point K/V vectors into each cache slot. Phantom
writes a *tag* instead: who wrote the slot, for which position, and a hash of
the exact token prefix up to that position. Because the hash covers the whole
prefix, two slots are interchangeable if and only if they hold the same
chain hash -- which is precisely the rule prefix caching is allowed to rely on.

That gives an exact correctness oracle with zero floating-point noise:
  * a read of the right chain hash          -> correct (even if another
                                               request wrote it: legal sharing)
  * a read of an EMPTY slot                 -> unwritten/uninitialised read
  * a read of a different chain hash        -> corruption or a cross-request leak
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

VOCAB = 32_000
ROOT = 0  # chain hash of the empty prefix


def _h(*parts: int) -> int:
    b = b"".join(p.to_bytes(8, "little", signed=False) for p in parts)
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


def chain_step(prev: int, token: int) -> int:
    """Hash of (prefix ending in `prev`) + `token`."""
    return _h(prev, token)


def chain_of(tokens: list[int]) -> list[int]:
    """chain[p] identifies the exact prefix tokens[0..p]."""
    out, h = [], ROOT
    for t in tokens:
        h = chain_step(h, t)
        out.append(h)
    return out


def fold(acc: int, chain_hash: int) -> int:
    """Phantom 'attention': fold every slot the model reads into one value."""
    return _h(acc, chain_hash)


def next_token(attended: int) -> int:
    """Phantom 'LM head': deterministic next token from what was attended."""
    return _h(attended, 0xA11CE) % VOCAB


@dataclass(frozen=True, slots=True)
class Tag:
    writer: int  # request id that wrote this slot (diagnostics only)
    pos: int  # token position the writer meant to store here
    chain: int  # chain hash of the writer's prefix up to `pos`
