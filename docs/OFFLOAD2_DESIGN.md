# Phase 5, target 2: vLLM's OffloadingConnector

Target 1 (`SimpleCPUOffloadConnector`) came out clean. Target 2 is vLLM's
other, larger offloading system.

## What it is

`OffloadingConnector` moves KV between the GPU cache and offloaded media
(CPU memory, and through the tiering specs, files and remote stores). The
decisions are made on the scheduler side:

| Code | Lines | Decides |
|---|---|---|
| `kv_connector/v1/offloading/scheduler.py` | ~2,600 | which blocks to store, when; sync vs async loads; deferred stores; preemption flushes (`jobs_to_flush`); failed loads (`failed_recving`); resets |
| `kv_offload/cpu/manager.py` (`CPUOffloadingManager`) | ~520 | which CPU chunk holds what; lookup, prepare/complete load and store; eviction (LRU); store threshold |

Interesting differences from target 1:

- **chunks**: a CPU chunk can hold several GPU blocks (`blocks_per_chunk`),
  and a transfer may enter its first chunk part-way
- **sync loads**: a load can be "expedited" and waited on within the step,
  because its request runs in that same step
- **stores are deferred** to the start of the next step
- **failures are part of the API**: a transfer can report
  `success=False`; a failed load must lead to recovery, and a failed store
  must never be served

## What runs unmodified, what is replaced

vLLM's offloading API is pluggable: an `OffloadingSpec` provides the
scheduler-side manager and the worker-side `OffloadingWorker`.
`PhantomCPUOffloadingSpec` (phantomkv/vllm_offload2.py) **is** vLLM's
`CPUOffloadingSpec`, same manager, same chunk sizing, with one override:
`get_worker()` returns `PhantomOffloadWorker`, tensor copies instead of CUDA
streams. It is selected with vLLM's documented out-of-tree mechanism
(`spec_name` + `spec_module_path` in `kv_connector_extra_config`); no vLLM
code is patched.

The copy engine keeps the lessons of target 1: loads and stores on separate
FIFO queues with separate delays (a load can overtake a slower store), a
copy reads its source when it executes, CPU chunks start as zeros, and
`wait()` completes the waited job and everything queued before it in that
stream. The chunk mapping is pure Python and unit-tested
(`phantomkv/offload_map.py`).

## Profiles

| Profile | Stresses |
|---|---|
| `offload2` | stores and loads of evicted prefixes |
| `offload2-chunk2` | 2 GPU blocks per chunk: partial first chunks, alignment |
| `offload2-slow-stores` | stores 40 steps, loads 1 |
| `offload2-preempt` | 24 GPU blocks, 8 sequences: preemption flushes |
| `offload2-small` | 32 GPU blocks, 32 CPU chunks: CPU-side eviction |
| `offload2-sync` | 1 MiB per-step budget: synchronous (expedited) loads |

## Faults (`PHANTOM_OFFLOAD2_FAULT`)

| Fault | Kind | Expected |
|---|---|---|
| `store-reported-early` | planted bug in our copy engine | caught (validation) |
| `load-fails` | environment failure vLLM must handle: every 3rd load fails | outputs stay correct (vLLM recovers) |
| `store-fails` | environment failure: every 3rd store fails | outputs stay correct (failed chunks never served) |

`load-fails` and `store-fails` are not bugs we planted in vLLM: they are
failures real storage can produce, and the API says vLLM handles them. A
violation there, on unmodified vLLM, would be a real finding (after ruling
out the copy engine).

## Status

| Day | Result |
|---|---|
| 19 | all five profiles clean and deterministic; planted fault caught 4/10; failed loads and failed stores handled correctly by vLLM; in CI |
| 20 | synchronous loads; partial-chunk revisits |
