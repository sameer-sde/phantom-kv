# Phantom — Deterministic Simulation Testing for LLM-Serving KV Caches

[![CI](https://github.com/sameer-sde/phantom-kv/actions/workflows/ci.yml/badge.svg)](https://github.com/sameer-sde/phantom-kv/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)](#quick-start)
[![Tests](https://img.shields.io/badge/tests-9%20passing-success)](#testing)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A testing framework that finds **cross-request data leaks and KV-cache
corruption** in LLM inference engines such as vLLM, SGLang and LMCache.
It does this **exactly** (hash equality, not floating-point comparison),
**on a laptop CPU** (no GPU, no model weights), and **reproducibly** (one
seed replays the entire run, event for event).

> Phantom is a model with no weights. To the serving engine it looks like a
> real LLM: it is scheduled, batched, preempted, prefix-cached, and its KV
> cache is shipped between machines. But instead of K/V vectors it writes a
> **provenance tag** into every cache slot, so any slot read by the wrong
> request is caught and the request that wrote it is named.

---

## Why this project

LLM servers squeeze throughput out of GPUs by **sharing memory between
users**: paged KV blocks, prefix caching, preemption, CPU offloading, and
**prefill/decode (PD) disaggregation**, where one machine computes the
prompt's KV cache and streams it to another with one-sided RDMA writes.
When that bookkeeping is wrong, nothing crashes. A user silently gets
corrupted output, or **another user's data**.

These bugs are hard to catch today:

| Problem | Why existing testing struggles |
|---|---|
| **Noisy oracle** | Comparing outputs on real GPUs is ambiguous: a *legitimate* prefix-cache hit already perturbs floating-point numerics, so "different output" ≠ "bug". |
| **Rare interleavings** | The dangerous cases (a client aborts *while* a KV write is in flight) depend on exact timing and almost never reproduce. |
| **Cost** | GPU fuzzing campaigns need expensive hardware and run slowly. |

Phantom addresses all three with two ideas from the distributed-systems
world:

- **An exact oracle.** Provenance tags + prefix chain hashes turn "is this
  KV correct?" into an equality check.
- **Deterministic simulation testing**, the technique behind FoundationDB,
  TigerBeetle's VOPR and Antithesis. A single PRNG seed controls every
  arrival, abort, network delay and memory-pressure event, so any failure
  replays exactly.

---

## Architecture

```mermaid
flowchart TD
    Seed(["seed: u64"])

    subgraph SIM ["Deterministic Simulator — sim.py"]
        RNG["Single PRNG<br/>random.Random(seed)"]
        WL["Workload generator<br/>arrivals · shared prefixes<br/>prompt lengths · aborts"]
        EQ["Event queue<br/>virtual clock<br/>heap ordered by (time, seq)"]
        NET["Simulated KV network<br/>random delay per block<br/>one-sided writes"]
        RNG --> WL --> EQ
        RNG --> NET
    end

    subgraph ENG ["Serving engine under test — engine.py"]
        SCH["Scheduler<br/>continuous batching<br/>admission · preemption"]
        KVM["KV block manager<br/>refcounts · free list<br/>LRU evictable list"]
        PC["Prefix cache<br/>chain-hash → block"]
        MEM[("Paged KV memory<br/>slot = Tag | empty")]
        SCH --> KVM
        KVM <--> PC
        KVM --> MEM
    end

    subgraph PH ["Phantom model — provenance.py"]
        W["write: Tag(writer, pos, chain_hash)"]
        R["read: every context slot<br/>via the engine's block table"]
        O{"chain_hash ==<br/>expected?"}
        R --> O
    end

    Seed --> RNG
    EQ -->|"submit / abort"| SCH
    NET -->|"remote prefill KV lands"| MEM
    SCH -->|"forward step"| W
    W --> MEM
    MEM --> R
    O -->|"yes"| TOK["next token<br/>H(fold of read hashes)"]
    O -->|"no"| V["Violation<br/>cross-request-leak · stale-read<br/>unwritten-read · block-leak · liveness"]
    V --> TR["Trace + fingerprint<br/>phantomkv replay --seed N"]
```

**Request flow:** the simulator submits a request → the scheduler admits it
(reusing prefix-cache blocks where the chain hash matches) or, for a PD
request, allocates destination blocks and lets the remote prefill stream
KV into them → each forward step, the Phantom model writes tags for new
positions and reads back **every** context slot through the engine's block
table → any mismatch is recorded with the full event trace.

---

## The oracle

Every request has a chain hash per position:
`chain[p] = H(chain[p-1], token[p])`, so `chain[p]` uniquely identifies
the prefix `tokens[0..p]`. The Phantom model writes
`Tag(writer, pos, chain[pos])` into the slot the engine assigns to `pos`.
When it reads the context back:

| What the slot holds | Verdict |
|---|---|
| same chain hash, **any** writer | ✅ correct, including legal prefix sharing between requests |
| nothing (never written) | ❌ `unwritten-read` |
| different chain hash, **another** request | ❌ `cross-request-leak` |
| different chain hash, same request | ❌ `stale-self-read` |

Two slots are interchangeable **iff** their chain hashes match. That is
exactly the rule prefix caching is allowed to exploit, so correct sharing
passes and everything else is flagged, with **zero false positives by
construction**. The next token is a hash of everything read, so a bad read
also changes the visible output, as it would with a real model.

---

## Core mechanisms

| Mechanism | What it does | Where |
|---|---|---|
| **Provenance tags** | Replace K/V tensors with `(writer, pos, chain_hash)`; BLAKE2b-64 chain over the prefix | `phantomkv/provenance.py` |
| **Paged KV block pool** | Fixed-size blocks, refcounts, free list, LRU eviction of unreferenced cached blocks | `phantomkv/engine.py` |
| **Prefix caching** | Full blocks indexed by the chain hash at the block boundary; hits always leave ≥ 1 token to compute so shared blocks are never written | `phantomkv/engine.py` |
| **Recompute preemption** | Under memory pressure the newest running request is evicted and later recomputed | `phantomkv/engine.py` |
| **PD disaggregation** | Decode allocates blocks; remote prefill streams one-sided writes per block with random delay. Writes don't check ownership, just like RDMA | `phantomkv/engine.py`, `phantomkv/sim.py` |
| **Abort handling** | Aborts at any state; blocks with in-flight writes are freed only after the last write lands | `phantomkv/engine.py` |
| **Deterministic simulator** | One seed → arrivals, prompts, shared prefixes, aborts, network delays; virtual clock; trace fingerprint | `phantomkv/sim.py` |
| **Liveness + leak checks** | Requests that never finish; blocks still referenced after quiescence | `phantomkv/sim.py`, `phantomkv/engine.py` |
| **Planted bugs** | Switch on known bug classes to prove the oracle catches them before targeting real engines | `--plant` flag |

---

## The bug class Phantom is built for

The abort-during-transfer race in PD disaggregation, the same class SGLang
disclosed in its
[October 2026 security update](https://www.lmsys.org/blog/2026-10-26-cf-security/):

```mermaid
sequenceDiagram
    participant C as Client A
    participant P as Prefill engine
    participant D as Decode engine
    participant B as Request B

    D->>D: allocate block 4 for A
    P-->>D: RDMA write (A, block 4) — in flight
    C->>D: abort A
    Note over D: BUG: free block 4 immediately
    B->>D: admitted → gets block 4, writes its own KV
    P-->>D: A's late write lands in block 4
    B->>D: forward step reads block 4
    Note over B: Phantom: cross-request-leak<br/>B read KV written by A
```

**Correct behaviour:** blocks with outstanding writes are released only when
the last write lands (or is cancelled and acknowledged).

---

## Validation

Before Phantom is trusted on real engines, it must (1) never raise a false
alarm on a correct engine and (2) catch known bug classes. Measured on a
MacBook-class CPU, 1,000 seeds per row, 40 requests per seed with mixed
local/PD traffic, shared prefixes and 25% client aborts:

| Engine configuration | Failing seeds | Violations reported |
|---|---|---|
| Correct engine | **0 / 1,000** | none (no false positives) |
| Planted `abort-free-race`: free blocks on abort while KV writes are in flight | **544 / 1,000** | `cross-request-leak` |
| Planted `stale-prefix-cache`: evict a cached block but keep its hash entry | **1,000 / 1,000** | `cross-request-leak`, `stale-self-read` |

| Metric | Value |
|---|---|
| Throughput | **~100 seeds/s** (≈ 4,000 simulated requests/s) on one CPU core |
| Determinism | same seed → identical trace fingerprint (tested) |
| GPU required | none |

---

## Proof: a real failure, captured end-to-end

```
$ python -m phantomkv fuzz --seeds 1000 --plant abort-free-race
phantomkv: 1000 seeds in 10.8s (93 seeds/s), plant=abort-free-race
544 failing seeds; violation kinds: {'cross-request-leak': 1107}
first failure: seed 0
  [t=8] cross-request-leak: req 8 pos 4: read KV of req 9 pos 0 (prefix differs)
replay with:  python -m phantomkv replay --seed 0 --plant abort-free-race
```

```
$ python -m phantomkv replay --seed 0 --plant abort-free-race
...
t=7     abort req 9 (was waiting_kv, inflight=1)     # client disconnects mid-transfer
...                                                  # buggy engine frees req 9's blocks
t=8     kv-land req 9 blk#0->4 left=0                # req 9's late write lands in block 4
t=8     VIOLATION [t=8] cross-request-leak: req 8 pos 4: read KV of req 9 pos 0 (prefix differs)

seed=0 fingerprint=7994438c2a79c6d6 ticks=76 finished=33 aborted=7 violations=2
```

Same seed, same fingerprint, on any machine.

---

## Quick start

### Requirements

- Python 3.10+
- No third-party dependencies (pytest for tests)

### Install

```bash
git clone https://github.com/sameer-sde/phantom-kv.git
cd phantom-kv
pip install -e ".[dev]"
```

### Run

```bash
# Correct engine: expect "no violations found"
python -m phantomkv fuzz --seeds 1000

# Prove the oracle catches a known bug class
python -m phantomkv fuzz --seeds 1000 --plant abort-free-race

# Replay one failing seed and print the story leading to the violation
python -m phantomkv replay --seed 0 --plant abort-free-race

# Heavier fault mix
python -m phantomkv fuzz --seeds 500 --remote-frac 0.9 --abort-frac 0.5
```

### CLI reference

| Command | Flags |
|---|---|
| `fuzz` | `--seeds N` · `--start S` · `--plant {abort-free-race, stale-prefix-cache}` · `--remote-frac F` · `--abort-frac F` · `--stop-first` |
| `replay` | `--seed N` · `--plant …` · `--context K` (trace lines before the violation) · `--full` |

---

## Testing

```bash
pytest -q
```

| Test | Guarantees |
|---|---|
| `test_no_false_positives_on_correct_engine` | 300 seeds, all requests finish or abort, zero violations |
| `test_no_false_positives_under_heavy_pressure` | 12 blocks, 80% PD, 50% aborts: still zero violations |
| `test_prefix_cache_is_exercised` | the workload actually produces prefix-cache hits (no vacuous passes) |
| `test_planted_bug_is_caught[*]` | every planted bug class is detected |
| `test_abort_race_is_reported_as_cross_request_leak` | the race is classified correctly |
| `test_same_seed_same_run` / `test_different_seeds_differ` | determinism |

CI runs the test suite on Python 3.10 and 3.12 plus a 1,000-seed clean fuzz on every push.

---

## Roadmap

| Phase | Status |
|---|---|
| 0. Exact oracle + deterministic simulator, validated on a toy engine | ✅ done |
| 1. Test-case shrinking: failing seed → minimal event trace | 🔄 next |
| 2. Phantom as an out-of-tree **vLLM** model on the CPU backend, writing tags through vLLM's real block tables and slot mappings | ⏳ |
| 3. Seeded fault injection against vLLM's real `EngineCore` | ⏳ |
| 4. Two-engine PD disaggregation with transfer faults (delay, reorder, duplicate, abort mid-transfer, handshake failure) | ⏳ |
| 5. Bug-hunting campaigns on vLLM, SGLang, LMCache, llm-d; issues and fix PRs upstream | ⏳ |

Details: [docs/ROADMAP.md](docs/ROADMAP.md) · design: [docs/DESIGN.md](docs/DESIGN.md) · daily progress: [docs/DEVLOG.md](docs/DEVLOG.md)

---

## Project structure

```
phantom-kv/
├── phantomkv/
│   ├── provenance.py      # Tag, chain hashes, phantom attention + LM head
│   ├── engine.py          # ToyEngine: paged KV, prefix cache, preemption, PD, aborts, oracle
│   ├── sim.py             # Deterministic simulator, workload generator, Result/fingerprint
│   ├── cli.py             # fuzz / replay commands
│   └── __main__.py        # python -m phantomkv
├── tests/
│   └── test_oracle.py     # no false positives · planted bugs caught · determinism
├── docs/
│   ├── DESIGN.md          # oracle, determinism, mapping to vLLM
│   ├── ROADMAP.md         # phases and exit criteria
│   └── DEVLOG.md          # daily engineering log
├── .github/workflows/ci.yml
└── pyproject.toml
```

---

## References

- Kwon, W., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM). SOSP 2023. https://arxiv.org/abs/2309.06180
- Zheng, L., et al. (2024). SGLang: Efficient Execution of Structured Language Model Programs. https://arxiv.org/abs/2312.07104
- Zhou, J., et al. (2021). FoundationDB: A Distributed Unbundled Transactional Key Value Store. SIGMOD 2021. Simulation testing: https://apple.github.io/foundationdb/testing.html
- TigerBeetle VOPR simulator. https://github.com/tigerbeetle/tigerbeetle
- Kingsbury, K. Jepsen: distributed systems safety research. https://jepsen.io
- Meng, R., et al. (2023). Greybox Fuzzing of Distributed Systems (Mallory). ACM CCS 2023. https://arxiv.org/abs/2305.02601
- GRIEF: Continuous Discovery of Vulnerabilities in LLM Serving Systems with Fuzzing (2026). https://arxiv.org/abs/2605.11202
- Gu, Y., et al. (2026). Ekka: Automated Diagnosis of Silent Errors in LLM Inference. ICML 2026.
- LMSYS (2026). SGLang security update: improving request isolation. https://www.lmsys.org/blog/2026-10-26-cf-security/

---

## Author

**Sameer Ahmed** · B.E. CSE, Osmania University ·
[GitHub](https://github.com/sameer-sde)

Related work: [vLLM PR #54832](https://github.com/vllm-project/vllm/pull/54832) ·
[RaftKV](https://github.com/sameer-sde/raftKv) ·
[Sentinel](https://github.com/sameer-sde/sentinel) ·
[Vektr](https://github.com/sameer-sde/vektr)

License: [MIT](LICENSE)
