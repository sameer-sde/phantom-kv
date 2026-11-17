# A model with no weights found two real bugs in LMCache

*Testing the KV caches of LLM servers, exactly, on a laptop CPU. Both bugs were already reported upstream; Phantom found them independently and tested the open fixes.*

**Sameer Ahmed** · November 2026 · [github.com/sameer-sde/phantom-kv](https://github.com/sameer-sde/phantom-kv)

---

When the KV cache of an LLM server goes wrong, nothing crashes. A request
reads a block that now belongs to someone else, and the model keeps
generating. The output is a little off, or it contains another user's
data. Or the cache quietly stops working, every prompt is recomputed, and
the only symptom is a bill.

I spent two and a half weeks building **Phantom**, a tool that tests this
memory exactly, on a laptop, with no GPU and no model weights. Pointed at
real vLLM and LMCache, it independently found two real, still-open LMCache
bugs. Both turned out to be already reported upstream, so the interesting
part is what came next: Phantom shrank one of them to a 7-event
reproduction, and showed that the fixes proposed for both actually work.

This post is about how it works and what it found.

## Why this memory is hard to test

LLM servers get their throughput by sharing GPU memory between users:

- **Paged KV blocks.** The cache is cut into fixed-size blocks, handed out
  and recycled like pages in an OS.
- **Prefix caching.** Two requests that start with the same tokens share
  the same blocks.
- **Preemption.** Under memory pressure a request is evicted and recomputed
  later.
- **Offloading.** Blocks are copied to CPU memory or to a separate cache
  server (such as LMCache) and loaded back later.
- **Prefill/decode disaggregation.** One machine computes the prompt's KV
  and streams it to another.

Every one of these is bookkeeping: who owns block 412 right now? Testing
that bookkeeping is hard for three reasons.

1. **No clean oracle.** On a GPU, a legitimate prefix-cache hit already
   changes floating-point results slightly, so "the output changed" does
   not mean "there is a bug".
2. **Rare timing.** The dangerous cases, such as a client aborting while
   its KV is still being written, depend on exact interleavings that
   almost never reproduce.
3. **Cost.** Fuzzing on GPUs is slow and expensive.

## The idea: a model that writes name tags instead of vectors

Phantom is a model with no weights. To the serving engine it looks like a
normal LLM: it is scheduled, batched, preempted, prefix-cached, offloaded.
But in every KV slot it is asked to fill, it writes a **provenance tag**
instead of K/V vectors:

```
Tag(writer = request id, position, chain_hash)
chain[p] = H(chain[p-1], token[p])
```

`chain[p]` identifies the exact prefix `tokens[0..p]`. When the model reads
its context back, each slot is checked:

| The slot holds | Verdict |
|---|---|
| the same chain hash, written by anyone | correct (this is legal prefix sharing) |
| nothing | `unwritten-read` |
| a different chain hash | wrong block, and the tag names who wrote it |

Two slots are interchangeable exactly when their chain hashes match, which
is precisely the rule prefix caching is allowed to exploit. Correct sharing
passes and anything else is flagged, so there are **no false positives by
construction**. The next token is a hash of everything read, so a bad read
also changes the visible output, the way it would with a real model.

## Deterministic simulation testing

The second idea comes from distributed systems (FoundationDB, TigerBeetle,
Antithesis): **one seed controls everything**. A seed becomes an explicit
list of events: which requests arrive, with which shared prefixes, at which
step, and which ones the client aborts. The same seed replays the same run,
event for event.

That gives two things for free:

- **Replay.** A failure on seed 9 is a failure on seed 9 on any machine.
- **Shrinking.** Delta debugging (ddmin) deletes events while the failure
  remains, and a 50-event run becomes a 6-event reproduction.

I first built this against a small simulated engine and validated the
oracle before trusting it anywhere else. Over 1,000 seeds: a correct engine
gave **0 failures**; a planted "free blocks on abort while writes are in
flight" bug failed **545**; a planted stale prefix-cache entry failed all
**1,000**. That runs at about 100 seeds per second on one core.

## Inside real vLLM

Then Phantom moved into the real thing: an out-of-tree model plugin for
vLLM's CPU backend. vLLM's own scheduler, KV-cache manager, prefix cache
and block tables decide where every token goes; Phantom writes its tags
there and re-reads the whole context through vLLM's block table at every
step. vLLM's source is not modified.

Two details mattered more than I expected:

- **Step by step, in process.** vLLM normally runs its engine core in a
  separate busy-looping process, so "abort at step 40" can land a step
  late. Running it in-process and calling `step()` myself made every run
  replayable.
- **Proof the path ran.** Every run prints vLLM's own counters (prefix
  cache hits, preemptions, chunked-prefill continuations, offloaded
  blocks). A clean result means nothing if the code under test never ran.

To show it catches real failures, I planted known bug classes at runtime.
A stale prefix-cache index entry was caught on **15 of 20 seeds**, and the
shrinker reduced one failure from 38 events to 13 by re-running real vLLM
235 times. With two vLLM engines exchanging KV through vLLM's connector API,
planted transfer faults (another request's KV, a load reported done but
never written, a duplicate late copy) were caught on 184 to 199 of 200
seeds, with **0 of 200** on the unmodified engines.

One finding from this phase stuck with me. A planted refcount bug never
reached Phantom: vLLM's own assertion `ref_cnt == 0` fired first. **Bugs
that make the bookkeeping inconsistent trip assertions. Bugs that leave it
consistent but wrong, like a stale index entry, trip nothing.** That second
kind is what Phantom is for.

## Hunting where nobody planted anything

The last phase pointed Phantom at code with no planted bugs.

**vLLM's two CPU offloading connectors** (`SimpleCPUOffloadConnector` and
`OffloadingConnector`), each run with its own code unmodified and only the
CUDA copy engine replaced. For `OffloadingConnector` alone, a 50-seed
campaign over eight configurations stored 21,181 blocks and loaded 2,243
back, every one verified, and injected storage failures were recovered
correctly. **No bugs.** That is a real result too: the
planted faults were caught, so the silence means something.

**LMCache**, a separate KV cache server, run CPU-only next to vLLM: 11,040
tokens served from LMCache, every one verified, and a bug planted inside
LMCache's own scatter code caught on 10 of 10 seeds. Then two things
turned up that nobody had planted.

### Bug 1: a fast restart is never noticed

I added a fault: restart the LMCache server between runs. Count the stores
per server lifetime:

| Restart | Stores per server lifetime |
|---|---|
| server down 25 s | 19, 20, 21, 6 |
| server back within a second | 19, **0, 0, 0** |

After a fast restart, vLLM never re-registers. Outputs stay correct (vLLM
recomputes), but caching is gone until vLLM restarts. The cause: the vLLM
side re-registers only after a heartbeat ping *fails*, and the server's
ping answered "fine" even for a worker it no longer knew. A restart shorter
than one 10-second heartbeat is invisible.

Before writing anything up, I searched LMCache's tracker. It was already
reported (issue #4642), with an open fix (PR #6106) that makes the ping
answer "not registered". So instead of filing a duplicate, I tested the
fix. With a fast restart followed by a full heartbeat:

| LMCache | Stores per server lifetime |
|---|---|
| without the fix | 19, 0, 0, 0 |
| with PR #6106 | 19, 20, 21, 19 |

One more thing fell out of this. On the older commit the fix is built on,
the same restart did not just lose the cache: **vLLM hung forever**. The
server rejected a store from the forgotten worker but, in that version,
sent no reply. Newer LMCache replies with an error. Same root cause, much
worse symptom, and a lesson for my harness: every trial now has a time
limit, because a hang is a result, not a stuck terminal.

### Bug 2: "harmless" nondeterminism that was a leak

Phantom's determinism check runs every seed twice and expects identical
runs. With LMCache, three seeds out of ten were not identical. No output
was wrong, so it would have been easy to call it noise.

LMCache's `/status` endpoint reports how many cached objects are
read-locked. After every request had finished, some still were: 1 to 5
per seed, accumulating, and a normal clear skips locked objects. The seeds
that differed on replay were exactly the ones whose replay added new
locks. With client aborts turned off: **0 locks, every seed identical**.

A single aborted request never leaked, at any moment I tried. So I let
Phantom find the combination: the same ddmin shrinker, with a new failure
test, "this run left read locks behind". It took a 41-event seed down to
**7 events** in 171 runs, leaking 3 locks on 5 of 5 re-runs. Logging the
connector's state at the abort showed the mechanism:

| At the abort | |
|---|---|
| vLLM request state | waiting |
| LMCache lookup | finished: 96 tokens found, 3 chunks read-locked |
| result consumed by the scheduler | not yet (the engine was busy) |
| locks after the abort | 3, never released |

The lookup had finished and locked the chunks, but vLLM had not picked up
the result yet. The abort path dropped the lookup on the vLLM side and
never told the server to release the locks. They expire after 300 s, and
until then those objects can be neither cleared nor evicted.

Again, it was already known (issue #5656), with an open fix (PR #6540).
Again I tested it, on Phantom's saved minimal cases and on the original
seeds:

| LMCache | locks left (two minimal cases) | seeds differing on replay |
|---|---|---|
| without the fix | 2 and 3 | 3 of 10 |
| with PR #6540 | **0 and 0** | **0 of 10** |

I posted both A/B results on the fix PRs.

## What I took away

- **Determinism checks find bugs that produce no wrong output.** The lock
  leak never corrupted a token. It showed up only because a replay was not
  identical.
- **A shrinker with a custom failure test is a debugger.** I did not guess
  the 7-event case; ddmin found it.
- **Search before you file.** Both bugs were already reported. The useful
  contribution was independent evidence that the open fixes work.
- **Measure that the path ran.** A clean run with zero prefix-cache hits
  tests nothing.

## Limits

Phantom tests bookkeeping, not numerics: it cannot find a wrong attention
kernel. Everything here ran on vLLM's CPU backend with a CPU-only LMCache
server, so GPU-only paths (CUDA copy streams, NIXL/RDMA transfers) are not
covered, and neither are SGLang or llm-d yet.

## Try it

```
git clone https://github.com/sameer-sde/phantom-kv && cd phantom-kv
python -m phantomkv fuzz --seeds 1000 --plant abort-free-race
python -m phantomkv shrink --seed 0 --plant abort-free-race
```

The simulator needs nothing but Python 3.10. The real-vLLM and LMCache
runs, the design notes and a day-by-day devlog are in the
[repository](https://github.com/sameer-sde/phantom-kv).

*I built Phantom with help from an AI coding assistant for code and log
analysis. I ran every experiment myself and checked the results.*
