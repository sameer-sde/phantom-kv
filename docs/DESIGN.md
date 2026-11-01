# Design notes

## 1. The oracle

Each request has a chain hash per position: `chain[p] = H(chain[p-1], token[p])`.
`chain[p]` identifies the exact prefix `tokens[0..p]`.

The Phantom model writes `Tag(writer, pos, chain[pos])` into the KV slot the
engine assigns to position `pos`. On every forward step it reads all context
slots through the engine's block table and checks `tag.chain == chain[p]`.

| Read | Meaning |
|---|---|
| matching chain hash, any writer | correct (legal prefix sharing) |
| `None` | unwritten slot read |
| different chain, different writer | **cross-request leak** |
| different chain, same writer | stale self-read (e.g. lost write after preemption) |

The next token is `H(fold(chain hashes read))`, so a wrong read also changes
the visible output, just like a real model would.

Why hashes and not floats: on real GPUs a legitimate cache hit can perturb
numerics, so output diffs can't separate bugs from noise. Hash equality can.

## 2. Determinism

One `random.Random(seed)` drives arrivals, prompt contents, shared prefixes,
abort times and network delays. Time is a virtual tick counter; events are a
heap ordered by `(time, insertion_seq)`. Every run has a trace and a
fingerprint (sha256 of the trace); tests assert same seed ⇒ same fingerprint.

## 3. What the toy engine models

* paged block pool with refcounts, free list, LRU evictable list
* prefix cache keyed by chain hash at block boundaries
* recompute preemption under memory pressure
* PD disaggregation: decode allocates blocks, remote prefill streams one-sided
  writes per block with random delay; the write does not check ownership (RDMA)
* aborts in every state; correct engine defers freeing until in-flight writes land

## 4. Mapping to vLLM (phase 2)

* register an out-of-tree model with `ModelRegistry`
* run on the CPU platform; KV tensors become int64 tag storage
* the model's attention reads slots via the real `slot_mapping` / block tables
  from the forward context — the engine's bookkeeping is under test, not ours
* open questions: tensor layout per attention backend; CUDA-graph-free path; hybrid KV groups
