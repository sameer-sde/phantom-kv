# Phase 5, target 1: vLLM's CPU KV offloading

Phases 2-4 proved that Phantom catches bugs, by planting known ones. Phase 5
points it at vLLM code where nobody planted anything.

## Why this target

`SimpleCPUOffloadConnector` keeps a second KV cache in CPU memory. Full GPU
blocks are copied there ("stores"); a later request whose prefix was evicted
from the GPU prefix cache copies it back ("loads") instead of recomputing it.

All the decisions are made by the scheduler-side manager,
`vllm/v1/simple_kv_offload/manager.py` (~1,500 lines at the pinned commit):

- which GPU blocks to store, eagerly as they fill or lazily just before the
  GPU cache would evict them (`lazy_offload`)
- which CPU block each one goes to; CPU-side LRU eviction
- which CPU blocks a request may load, and pinning them until it does
- holding GPU blocks until their store completes
- preemption (`need_flush`), aborts, `reset_cache`, and partial tail blocks

This is exactly the "consistent but wrong" bookkeeping Phantom is built for:
if the manager loads the wrong CPU block, a stale one, or one whose store
has not completed, vLLM's own state stays consistent and nothing asserts,
but a request silently reads someone else's KV.

## What runs unmodified, and what is replaced

| Part | Here |
|---|---|
| `SimpleCPUOffloadConnector` | vLLM's, unmodified (selected by name) |
| scheduler-side manager, CPU block pool, LRU, pinning, events | vLLM's, unmodified |
| scheduler, GPU block pool, prefix cache | vLLM's, unmodified |
| copy engine (`SimpleCPUOffloadWorker`: CUDA streams + DMA) | **replaced** by `PhantomOffloadWorker` |

The CPU backend has no CUDA streams, so the copy engine is the one piece
that cannot run. `PhantomOffloadWorker` (phantomkv/vllm_offload.py) executes
the same instructions, the block lists in `SimpleCPUOffloadMetadata`, with
plain tensor copies, and reports completions through the same channels:
`finished_recving` for loads (via `load_event_to_reqs`), worker metadata
(`completed_store_events`) for stores, a full flush on `need_flush`.

Faithfulness choices:

- a transfer reads its source **when it executes**, like DMA, not when it was
  requested; with `PHANTOM_OFFLOAD_DELAY=n` it executes n steps later, so a
  GPU block the manager releases too early can be overwritten in between
- transfers complete in issue order (the real streams are FIFO)
- the CPU cache starts as zeros, so loading a block nothing was ever stored
  into reads as `unwritten-read`
- between seeds the harness keeps stepping until every transfer has
  completed, so nothing lands in the next run

A violation therefore means the manager told the copy engine to do something
wrong, or the replaced copy engine is wrong. Before any finding is reported,
the copy engine has to be ruled out (minimal repro, then the manager's
decisions traced by hand).

## Profiles (`phantomkv vllm-fuzz --profile ...`)

| Profile | Stresses |
|---|---|
| `cpu-offload` | eager stores, loads of evicted prefixes (revisit wave) |
| `cpu-offload-lazy` | lazy stores: the GPU free-queue cursor |
| `cpu-offload-slow` | transfers take 2 steps: early release of GPU or CPU blocks |
| `cpu-offload-small` | 32 GPU + 32 CPU blocks: CPU-side LRU eviction |
| `cpu-offload-very-slow` | transfers take 25 steps: still in flight when their requests finish and the revisit wave arrives |
| `cpu-offload-preempt` | 24 GPU blocks, 8 sequences: preemption while transfers are in flight (`need_flush`) |

Evidence per campaign: store events / blocks stored, load events / blocks
loaded back, flushes. A campaign that never loads a block back did not test
the load path, and says so.

## Validation (planted, in our copy engine)

`PHANTOM_OFFLOAD_FAULT=store-reported-early` (with `cpu-offload-very-slow`): a
store is reported complete when it is issued, but the copy still runs 25
steps later. The manager, trusting the report, releases the GPU blocks and
publishes the CPU blocks: a GPU block reused in between is copied with the
wrong contents, and a load can read a CPU block before anything is in it.
Must be caught; unmodified runs must stay clean.

The first plant tried, `store-before-compute` (a store reading the GPU
blocks before this step's forward pass, the race vLLM's worker guards
against, #45704), fired 256 times and changed nothing: the manager only
stores blocks that were already full before the step, and the CPU backend
runs the forward pass synchronously. That race needs overlapped GPU
execution, so it cannot be reproduced here.

## Milestones

| Day | Milestone |
|---|---|
| 17 | offload running on the CPU backend; profiles; planted fault caught |
| 18+ | campaigns across all four profiles; any violation on unmodified vLLM is minimised and traced to the manager line responsible |
