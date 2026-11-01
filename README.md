# Phantom 👻

**Deterministic simulation testing for LLM-serving KV caches.**
Find cross-request data leaks and KV-cache corruption in inference engines
(vLLM, SGLang, LMCache) — exactly, on a laptop CPU, replayable from one seed.

> Status: **Day 1** — oracle + simulator validated on a toy engine.
> Next: plug the Phantom model into real vLLM. See [ROADMAP](docs/ROADMAP.md) and [DEVLOG](docs/DEVLOG.md).

---

## Why the name "Phantom"?

A phantom looks like it is there, but has no body. Phantom is a **model with no
weights**: to the serving engine it looks like a real LLM — it is scheduled,
batched, preempted, prefix-cached and its KV is shipped between machines — but
instead of floating-point numbers it writes a **provenance tag** into every KV
slot: *who wrote this, for which position, and a hash of the exact prompt
prefix*.

When the model later reads its context back, every slot must carry the hash of
*its own* prefix. Anything else is a bug — and the tag names the culprit:

```
VIOLATION cross-request-leak: req 8 pos 4: read KV of req 9 pos 0 (prefix differs)
```

## The problem

Modern LLM servers share and reuse GPU memory between users: paged KV blocks,
prefix caching, preemption, offloading, and **prefill/decode disaggregation**
where KV is streamed between machines with one-sided RDMA writes. When that
bookkeeping goes wrong, nothing crashes — a user silently gets corrupted output
or **another user's data**.

Testing this today is hard:

* Real-GPU output comparison is noisy: a legitimate prefix-cache hit already
  perturbs floating-point results, so a "different answer" is not proof of a bug.
* The bad interleavings (abort *while* a KV write is in flight) are rare and
  almost impossible to reproduce.

Phantom fixes both: **an exact oracle** (hashes, not floats) and **deterministic
simulation** (one PRNG seed controls arrivals, aborts and network delays).

## Quick start

```bash
pip install -e ".[dev]"
pytest                                             # oracle self-tests

python -m phantomkv fuzz --seeds 500               # correct engine: no violations
python -m phantomkv fuzz --seeds 500 --plant abort-free-race
python -m phantomkv replay --seed 0 --plant abort-free-race
```

Example replay — the abort/in-flight-write race, told in four lines:

```
t=7  abort req 9 (was waiting_kv, inflight=1)    # client disconnects mid-transfer
...                                              # buggy engine frees req 9's blocks;
...                                              # req 8 is given block 4
t=8  kv-land req 9 blk#0->4 left=0               # req 9's late RDMA write lands in block 4
t=8  VIOLATION cross-request-leak: req 8 pos 4: read KV of req 9 pos 0
```

## How it works

```
 seed ──► Simulator (single PRNG, virtual clock, event queue)
              │ arrivals · aborts · network delays · memory pressure
              ▼
         Serving engine  (today: ToyEngine  →  next: real vLLM scheduler/KV manager)
              │ block tables · prefix cache · preemption · PD KV transfer
              ▼
         Phantom model   writes Tag(writer, pos, chain_hash) into each slot
                         reads all context slots → checks chain_hash == expected
              ▼
         Violations + full trace  →  replay / (soon) shrink to minimal repro
```

Oracle rule: two KV slots are interchangeable **iff** they hold the same chain
hash. That is exactly the rule prefix caching is allowed to exploit, so
legitimate sharing passes and everything else is caught.

## Validation so far

| Configuration | Seeds | Result |
|---|---|---|
| Correct engine (mixed local + PD, aborts, prefix cache) | 1,000 | **0 false positives** |
| Planted: free blocks on abort with KV writes in flight | 1,000 | caught in 544 seeds |
| Planted: evict cached block but keep stale hash entry | 1,000 | caught in 1,000 seeds |

The first planted bug models the class SGLang disclosed in its
[Oct 2026 security update](https://www.lmsys.org/blog/2026-10-26-cf-security/).

## Repository layout

```
phantomkv/provenance.py   tags, chain hashes, the phantom "attention" + "LM head"
phantomkv/engine.py       ToyEngine: paged KV, prefix cache, preemption, PD, aborts
phantomkv/sim.py          deterministic simulator + workload generator
phantomkv/cli.py          fuzz / replay
tests/                    oracle self-tests (no false positives, catches plants)
docs/                     design, roadmap, daily devlog
```

## Author

Sameer Ahmed — B.E. CSE, Osmania University.
Previously: vLLM PR #54832, Raft from scratch (RaftKV), 71K RPS inference serving (Sentinel).

License: MIT
