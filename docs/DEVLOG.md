# Devlog

One entry per working day: what changed, what I learned, what's next.

## Day 1 — 2026-11-01

**Built**
- Provenance oracle: tags + chain hashes; phantom attention/LM head.
- ToyEngine: paged KV, prefix cache (LRU), preemption, PD disaggregation, aborts.
- Deterministic simulator: one seed controls everything; trace + fingerprint.
- CLI: `fuzz`, `replay`. 9 tests.

**Results**
- Correct engine: 0 false positives over 1,000 seeds (~110 seeds/s on CPU).
- Planted `abort-free-race` (SGLang Oct-2026 class): caught in 544/1,000 seeds,
  reported as `cross-request-leak` with a 15-line replay.
- Planted `stale-prefix-cache`: caught in 1,000/1,000 seeds.

**Learned**
- Prefix-cache hits must leave ≥1 token to compute, otherwise the engine would
  write into a shared block.
- The correct abort rule for PD: blocks with in-flight writes can only be freed
  once the last write lands (or is cancelled and acknowledged).

**Next**
- Shrinker: reduce a failing seed's event list to a minimal reproduction.
- Read vLLM v1 `KVCacheManager`, `BlockPool`, and the simulate-forward PR (#47922).

## Day 2 — 2026-11-01

**Built**
- Simulator refactor: a run is now `(seed, events)`. `generate_workload()`
  turns a seed into an explicit list of `Event` records that can be edited.
- Stable network delays: `delay = 1 + H(seed, rid, block) % max_delay`
  instead of sequential PRNG draws.
- `shrink.py`: delta debugging (ddmin) + `shrink_seed()`; `phantomkv shrink` CLI.
- 9 new tests (18 total), including a check that every shrunk repro is 1-minimal.

**Results**
- 92 failing seeds across both planted bugs: ~50 events → median **6**, max **13**.
- ~0.2 s and ~100 simulations per shrink on one CPU core.
- Seed 0 of the abort race: 53 events → 6, in 29 runs / 0.06 s.

**Learned**
- Shrinking needs *stable* randomness. With sequential `rng.randrange()` for
  network delays, deleting one request shifted every later delay, so failures
  disappeared for the wrong reason. Hashing (seed, request, block) fixes it;
  FoundationDB and Hypothesis rely on the same idea.
- "Still fails" must mean the *same* violation kind, otherwise ddmin can
  wander from one bug to a different one.
- Python late binding: `lambda r=r: ...` inside loops, never `lambda: ... r`.

**Next**
- Phase 2: read vLLM v1 `KVCacheManager`, `BlockPool`, CPU attention backend
  and the simulate-forward PR (#47922); design the out-of-tree Phantom model.
