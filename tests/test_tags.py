"""Tag encoding must round-trip exactly through float32, the KV-cache dtype."""

import array

import pytest

from phantomkv import tags
from phantomkv.provenance import ROOT, chain_of, chain_step


def as_float32(vals):
    return list(array.array("f", vals))  # what a float32 KV slot would store


def test_roundtrip_through_float32():
    for pos, tok, chain in [(0, 0, 0), (31, 50_271, 2**64 - 1), (1 << 23, 12345, 0xDEADBEEFCAFEF00D)]:
        enc = tags.encode(pos, tok, chain)
        assert len(enc) == tags.TAG_WIDTH
        assert tags.decode(as_float32(enc)) == (pos, tok, chain)


def test_empty_slot_decodes_to_none():
    assert tags.decode([0.0] * tags.TAG_WIDTH) is None


def test_rejects_values_that_would_lose_precision():
    with pytest.raises(ValueError):
        tags.encode(1 << 24, 0, 0)


def test_expected_continuation_follows_the_chain():
    prompt, vocab = [5, 6, 7], 50_272
    out = tags.expected_continuation(prompt, 4, vocab)
    h = chain_of(prompt)[-1]
    for t in out:
        assert t == tags.next_token(h, vocab)
        h = chain_step(h, t)


def test_different_prompts_diverge():
    a = tags.expected_continuation([1, 2, 3], 8, 50_272)
    b = tags.expected_continuation([1, 2, 4], 8, 50_272)
    assert a != b and all(0 <= t < 50_272 for t in a + b)


def test_root_is_empty_prefix():
    assert chain_step(ROOT, 9) == chain_of([9])[0]
