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


def _missing_touch() -> None:
    """A prefix-cache hit on a block that another running request still owns
    does not count the new owner. When the first owner finishes, the block's
    refcount reaches 0 and it is recycled while the second request still
    reads it.

    Real code: vllm/v1/core/block_pool.py BlockPool.touch, which increments
    block.ref_cnt for every hit.
    """
    from vllm.v1.core.block_pool import BlockPool

    def touch(self, blocks) -> None:
        for block in blocks:
            if block.is_null:
                continue
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
                block.ref_cnt += 1
            # BUG (planted): a block already in use (ref_cnt > 0) is shared
            # with the new request without counting it as an owner.
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    BlockPool.touch = touch


def _abort_free_race() -> None:
    """A request aborted while its KV is still being loaded asynchronously has
    its blocks freed immediately, and the load's late completion is silently
    ignored. The in-flight write then lands in blocks that may already belong
    to another request: the SGLang Oct-2026 isolation bug class.

    Real code: vllm/v1/core/sched/scheduler.py. finish_requests() passes
    delay_free_blocks=True for WAITING_FOR_REMOTE_KVS requests, and
    _update_from_kv_xfer_finished() frees them when the load reports done.
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    orig_free = Scheduler._free_request
    orig_done = Scheduler._update_from_kv_xfer_finished

    def _free_request(self, request, delay_free_blocks: bool = False):
        # BUG (planted): ignore delay_free_blocks; free now even if a load is in flight
        return orig_free(self, request, delay_free_blocks=False)

    def _update_from_kv_xfer_finished(self, kv_connector_output):
        recv = kv_connector_output.finished_recving
        if recv:  # the freed request is gone; drop its completion quietly
            kv_connector_output.finished_recving = {r for r in recv if r in self.requests} or None
        return orig_done(self, kv_connector_output)

    Scheduler._free_request = _free_request
    Scheduler._update_from_kv_xfer_finished = _update_from_kv_xfer_finished


def _lmcache_chunk_rotate() -> None:
    """LMCache (target 3): when a retrieval scatters several chunks back into
    vLLM's paged KV, the chunk list is rotated by one, so every chunk lands in
    its neighbour's blocks. Planted in LMCache's own scatter path
    (EngineDrivenTransferContext._scatter_group), not in vLLM."""
    from lmcache.v1.client.worker_transfer import EngineDrivenTransferContext

    orig = EngineDrivenTransferContext._scatter_group

    def _scatter_group(self, kv_caches, gid, group_block_ids, chunks, first_chunk,
                       skip_first_n_tokens):
        if len(chunks) > 1:
            chunks = list(chunks[1:]) + [chunks[0]]  # BUG (planted)
        return orig(self, kv_caches, gid, group_block_ids, chunks, first_chunk,
                    skip_first_n_tokens)

    EngineDrivenTransferContext._scatter_group = _scatter_group


PLANTS: dict[str, Callable[[], None]] = {
    # bug class 1: hash -> block index keeps an entry for an evicted block
    "stale-prefix-cache": _stale_prefix_cache,
    # bug class 2: reference counting misses a sharer
    "missing-touch": _missing_touch,
    # bug class 3 (PD): blocks freed while an async KV load is still in flight
    "abort-free-race": _abort_free_race,
    # bug class 4 (LMCache): retrieved chunks scattered into the wrong blocks
    "lmcache-chunk-rotate": _lmcache_chunk_rotate,
}


def apply(name: str) -> None:
    if name not in PLANTS:
        raise ValueError(f"unknown vLLM plant {name!r}; choose from {sorted(PLANTS)}")
    PLANTS[name]()
