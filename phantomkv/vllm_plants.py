"""Known bug classes, planted into real vLLM at runtime (for validating Phantom).

A plant is applied only when the environment variable PHANTOM_VLLM_PLANT names
it. The plugin (vllm_plugin.py) runs in every vLLM process, including the
EngineCore that owns the scheduler and block pool, so the plant takes effect
where the bookkeeping actually happens. vLLM's source on disk is never modified.

Each plant replaces one vLLM method with a subtly broken copy of itself.
"""

from __future__ import annotations

from collections.abc import Callable


def _stale_prefix_cache() -> None:
    """Evicting a cached block forgets the block's own hash but leaves the entry
    in the hash -> block index. A later request with that prefix is handed a
    block that now holds someone else's KV.

    Real code: vllm/v1/core/block_pool.py BlockPool._maybe_evict_cached_block,
    which calls _remove_cached_block_hashes() to drop the index entries.
    """
    from vllm.v1.core.block_pool import BlockPool

    def _maybe_evict_cached_block(self, block) -> bool:
        had_hash = block.block_hash is not None or bool(
            self.cached_block_hashes_by_block.get(block.block_id))
        if not had_hash:
            return False
        self.cached_block_hashes_by_block.pop(block.block_id, None)
        block.reset_hash()
        # BUG (planted): missing self.cached_block_hash_to_block.pop(...)
        return True

    BlockPool._maybe_evict_cached_block = _maybe_evict_cached_block


PLANTS: dict[str, Callable[[], None]] = {
    "stale-prefix-cache": _stale_prefix_cache,
}


def apply(name: str) -> None:
    if name not in PLANTS:
        raise ValueError(f"unknown vLLM plant {name!r}; choose from {sorted(PLANTS)}")
    PLANTS[name]()
