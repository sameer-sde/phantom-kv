# Phase 2 design: Phantom inside real vLLM

**Pinned target:** vLLM `0.33.0rc1.dev485`, commit
[`0e5b2ba118`](https://github.com/vllm-project/vllm/commit/0e5b2ba118cfb3763a3640942e02dd8d93e8ed6c),
built from source with `VLLM_TARGET_DEVICE=cpu` on Apple Silicon (macOS 26,
Apple Clang 17, Python 3.12). All file:line references below are at this commit.

**Goal of Phase 2:** run vLLM's *real* scheduler, KV-cache manager, block pool,
prefix cache and model runner unmodified, with the Phantom model plugged in as an
out-of-tree model, so that any KV-cache bookkeeping bug in vLLM shows up as an
exact provenance violation, on a laptop CPU.

---

## 1. Code map: where the toy engine's concepts live in vLLM

| ToyEngine concept (`phantomkv/engine.py`) | vLLM equivalent | Where |
|---|---|---|
| `ToyEngine.step()` scheduling loop | `Scheduler.schedule()` | `vllm/v1/core/sched/scheduler.py:698` |
| preemption (`_preempt_for`) | `Scheduler._preempt_request()` | `scheduler.py:1798` |
| abort | `Scheduler.finish_requests()` → `_free_request()` | `scheduler.py:2954`, `:3021` |
| "defer free while writes in flight" | `delay_free_blocks` from the KV connector; `defer_block_free` + `deferred_frees` | `scheduler.py:3021`, `:3074`, `:168`, `:194` |
| prefix-cache lookup on admit | `KVCacheManager.get_computed_blocks()` | `vllm/v1/core/kv_cache_manager.py:266` |
| allocate blocks for new tokens | `KVCacheManager.allocate_slots()` | `kv_cache_manager.py:379` |
| release a request's blocks | `KVCacheManager.free()` | `kv_cache_manager.py:628` |
| free list / LRU evictable list | `FreeKVCacheBlockQueue` | `vllm/v1/core/kv_cache_utils.py:260` |
| block + refcount | `KVCacheBlock` | `kv_cache_utils.py:164` |
| chain hash at block boundary | `hash_block_tokens()`, `get_request_block_hasher()` | `kv_cache_utils.py:709`, `:955` |
| `_alloc()` (evict cached block on reuse) | `BlockPool.get_new_blocks()` → `_maybe_evict_cached_block()` | `vllm/v1/core/block_pool.py:706`, `:769` |
| register full block in cache | `BlockPool.cache_full_blocks()` | `block_pool.py:230` |
| prefix hit (refcount++) | `BlockPool.touch()` | `block_pool.py:792` |
| `_release()` | `BlockPool.free_blocks()` | `block_pool.py:816` |
| `r.blocks` → slot index | `BlockTable.compute_slot_mapping()` (worker side) | `vllm/v1/worker/block_table.py:153` |
| the model reading its context | `get_attention_context(layer_name)` → `(attn_metadata, layer, kv_cache, slot_mapping)` | `vllm/model_executor/layers/attention/attention.py:695` |
| PD transfer finishing | NIXL `request_finished()` | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:550` |

**Mechanisms vLLM has that the toy engine does not**, and so are high-value
targets for later phases:

* `defer_block_free` / `deferred_frees`: frees are postponed while an in-flight
  step may still write a block (`_request_blocks_can_be_freed` / `_free_request_blocks`, `scheduler.py:3074`).
* `BlockPool.unpin_blocks()` + reuse watchers: released blocks stay *readable*
  until reused (`block_pool.py:748`).
* Copy-on-write retained blocks (`_free_cow_retained_blocks`, `scheduler.py:3085`) and
  `is_block_writable()` (`block_pool.py:812`).
* Hybrid KV groups (`kv_cache_coordinator.py`, `single_type_kv_cache_manager.py`).
* `example_connector.py`: a filesystem KV connector we can use for PD
  disaggregation on CPU without NIXL/RDMA hardware (Phase 4).

## 2. How a forward step reaches the KV cache on CPU

1. The scheduler emits `SchedulerOutput` (new/cached requests, block ids per request).
2. The CPU model runner updates per-request `BlockTable` rows and computes the
   **slot mapping**: `slot = block_id * block_size + offset` for every new token.
3. Attention metadata for the CPU backend (`CPUAttentionMetadata`,
   `vllm/v1/attention/backends/cpu_attn.py:118`) carries `query_start_loc`,
   `seq_lens`, `block_table`, `slot_mapping`.
4. The layer's KV cache tensor has shape
   `[num_blocks, num_kv_heads, block_size, 2 * head_size]` (`cpu_attn.py:376`, `:423–427`):
   keys in the first `head_size` columns, values in the second.

Any layer can obtain all of this with `get_attention_context(layer_name)`.

## 3. The Phantom model

An out-of-tree model registered with
`ModelRegistry.register_model("PhantomForCausalLM", "phantomkv.vllm_model:PhantomForCausalLM")`
(`vllm/model_executor/models/registry.py:1203`).

**Structure**

* One real `vllm.model_executor.layers.attention.Attention` layer
  (1 head, 1 KV head, `head_size=32`, the smallest the CPU backend supports;
  `cpu_attn.py:77`). It exists so vLLM allocates and binds a real paged KV cache
  and builds real attention metadata for it. Phantom never calls its kernel.
* No weights: `load_weights()` is a no-op.
* Config: reuse a tiny HF config (`facebook/opt-125m`) with
  `hf_overrides={"architectures": ["PhantomForCausalLM"]}`.

**Forward pass** (per scheduled token, using `get_attention_context`):

1. *Write.* For each new token at position `p` with slot `s`, write a tag into
   `kv_cache[s // B, 0, s % B, 0:8]` (the key half):
   `[MAGIC, p, token, h0, h1, h2, h3, 0]`, where `h0..h3` are the 16-bit chunks
   of the 64-bit chain hash `chain[p] = H(chain[p-1], token)`. `chain[p-1]` is
   read back from the slot at `p-1`. Every value is an integer < 2^24, so it is
   **exact in float32**. KV dtype must be float32 (`dtype="float32"`).
2. *Read and check.* For each sequence, walk its `block_table` up to `seq_len`
   and verify, for every position `q`:
   * slot is not empty (`MAGIC` present) → else `unwritten-read`
   * `tag.pos == q` → else `misplaced-block`
   * `tag.chain == H(prev.chain, tag.token)` → else `chain-break`
     (catches foreign blocks spliced into the middle of a context)
3. *Emit.* `next_token = H(chain[last]) % vocab`. `compute_logits()` returns
   logits with a single large value at `next_token`, so greedy sampling
   (`temperature=0`) produces it deterministically.

**What the harness checks** (outside the model, pure Python):

* For every request, recompute the reference chain from its *true* prompt and
  the tokens vLLM returned. Any mismatch is a violation. This catches the one
  case in-model checks can't: a context that is internally consistent but
  belongs to another prompt (for example, the wrong first block).
* Violations raised inside the model are collected from a shared list; we run
  vLLM in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so the harness can read it.

**Engine settings for determinism and speed**

Two rules learned the hard way on macOS (Day 3):

1. **Always guard the entry point with `if __name__ == "__main__":`.** macOS
   starts subprocesses with `spawn`, which re-imports the script; without the
   guard, the EngineCore child tries to start its own engine and Python aborts
   with the "bootstrapping phase" RuntimeError.
2. **Always set `kv_cache_memory_bytes` explicitly.** By default the CPU worker
   reserves ~90% of RAM for the KV cache (`gpu_memory_utilization`, despite
   the name; see `vllm/v1/worker/cpu_worker.py:205–255`), which fails on a 16 GB
   laptop with other apps open. Phantom wants a small cache anyway, to force
   eviction and preemption.


```python
LLM(model="facebook/opt-125m",
    hf_overrides={"architectures": ["PhantomForCausalLM"]},
    dtype="float32", enforce_eager=True, skip_tokenizer_init=True,
    enable_prefix_caching=True, block_size=32,
    kv_cache_memory_bytes=<small, to force eviction and preemption>,
    max_num_seqs=<small>, seed=0)
```

Prompts are passed as token ids (`TokensPrompt`), so no tokenizer is needed.

## 4. Milestones

| Day | Deliverable | Done when |
|---|---|---|
| 3 | vLLM CPU build on Mac; this design | ✅ smoke test generates text (`scripts/vllm_smoke.py`) |
| 4 | `phantomkv/vllm_model.py`: register, allocate KV, no-op forward | ✅ `ARCH: ['PhantomForCausalLM']`, 4 tokens generated (`scripts/vllm_phantom_hello.py`) |
| 5 | Tag write + read via `get_attention_context` | ✅ 8 prompts × 2 rounds with prefix-cache hits: 748 tags written, 19,776 slot reads, 0 violations, all outputs match reference |
| 6 | Harness: seeded workload → step-by-step `LLMEngine` → reference check | ✅ done on Day 7: `phantomkv vllm-fuzz`, 20 seeds × 2 runs clean and deterministic |
| 7 | Plant a bug in vLLM (runtime plant: `_maybe_evict_cached_block` keeps a stale index entry) | ✅ done on Day 6: caught (96 `misplaced-block` violations, 7/8 wrong outputs); unmodified vLLM passes. Shrinking: Day 8 |
| 8 | Aborts, preemption pressure, prefix sharing at scale; shrinking on real vLLM | ✅ revisit wave: planted bug caught on 15/20 seeds; `vllm-shrink`: 38 → 13 events, reproduces 3/3 |

## 5. Risks and open questions

* **Block size.** The CPU kernel needs a multiple of 32 (`cpu_attn.py:64`). Smaller
  blocks would give more sharing and eviction per token; 32 is the floor.
* **Aborts at chosen steps**: solved on Day 7 by driving `LLMEngine` directly
  (`add_request` / `abort_request` / `step`).
* **Same-step prefix hits**: vLLM caches blocks at schedule time, so a request
  can hit blocks another request fills in the same forward pass. Phantom must
  write the whole batch before checking anything (fixed on Day 7).
* **torch.compile / graphs** must stay off (`enforce_eager=True`): the model
  has Python-side effects.
* **Async scheduling** may reorder when frees become visible; start with it off,
  then turn it on deliberately as a fuzz dimension.
* **Version drift.** vLLM moves fast, so pin the commit in CI and update deliberately.
