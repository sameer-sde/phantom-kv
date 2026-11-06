"""Block <-> chunk mapping used by the OffloadingConnector copy engine."""

import pytest

from phantomkv.offload_map import chunk_pairs


def test_one_block_per_chunk_is_one_to_one():
    assert list(chunk_pairs([7, 3, 9], [0, 5, 2], 0, 1)) == [(7, 0, 0), (3, 5, 0), (9, 2, 0)]


def test_two_blocks_per_chunk_fill_chunks_in_order():
    assert list(chunk_pairs([10, 11, 12, 13], [4, 6], 0, 2)) == [
        (10, 4, 0), (11, 4, 1), (12, 6, 0), (13, 6, 1)]


def test_first_chunk_entered_part_way():
    # logical blocks 3,4,5 with 2 blocks per chunk: chunk #0 holds logical 2-3
    # (enter at sub-block 1), chunk #1 holds logical 4-5
    assert list(chunk_pairs([20, 21, 22], [8, 9], 3, 2)) == [(20, 8, 1), (21, 9, 0), (22, 9, 1)]


def test_too_few_chunks_is_an_error():
    with pytest.raises(ValueError):
        list(chunk_pairs([1, 2, 3], [0], 0, 2))
