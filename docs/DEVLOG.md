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

## Day 3 — 2026-11-02

**Built**
- vLLM `0.33.0rc1.dev485` (commit `0e5b2ba118`) built from source for CPU on
  Apple Silicon (macOS 26, Apple Clang 17, Python 3.12, uv). Smoke test passes:
  `opt-125m` generates " the capital of the French Republic."
- `docs/VLLM_INTEGRATION.md`: code map from our ToyEngine to vLLM v1 with
  file:line references, and the design of `PhantomForCausalLM`.
- `scripts/vllm_smoke.py`.

**Debugged**
1. `RuntimeError: Engine core initialization failed`. The real error was
   hidden higher up: macOS uses `spawn` for subprocesses, which re-imports the
   script, so the EngineCore child re-ran top-level code. Fix: put everything
   under `if __name__ == "__main__":`.
2. `ValueError: Available memory ... (5.0/16.0 GiB) ... less than requested
   memory for kv (12.98/14.72 GiB)`. Running in-process
   (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) surfaced it. On CPU,
   `gpu_memory_utilization` reserves ~90% of *system RAM* for the KV cache.
   Fix: `kv_cache_memory_bytes=1 << 30`.

**Learned (vLLM internals)**
- `get_attention_context(layer_name)` gives any layer the attention metadata
  (block table, seq lens), the raw KV tensor and the slot mapping, so Phantom
  can plug in without patching vLLM.
- CPU KV layout: `[num_blocks, num_kv_heads, block_size, 2*head_size]`;
  block size must be a multiple of 32; head size >= 32.
- vLLM already has defences our toy engine only modelled: connector-driven
  `delay_free_blocks`, `defer_block_free`/`deferred_frees`, copy-on-write
  retained blocks, `unpin_blocks` reuse watchers. These newer paths are the
  bug-hunting targets.

**Next (Day 4)**
- `phantomkv/vllm_model.py`: register `PhantomForCausalLM`, allocate its KV
  cache, no-op forward; `LLM(...)` constructs with the Phantom architecture.

## Day 4 — 2026-11-02

**Built**
- `phantomkv/vllm_model.py`: `PhantomForCausalLM`, a weightless model with one
  real `Attention` layer (1 head, head size 32), so vLLM allocates and binds a
  paged KV cache for it. Forward/logits emit a placeholder token for now.
- `phantomkv/vllm_plugin.py` + `vllm.general_plugins` entry point: vLLM calls
  `register()` in every process it starts (main process, EngineCore, workers).
- `scripts/vllm_phantom_hello.py`.

**Result: first run of Phantom inside real vLLM** (vLLM `0.33.0rc1.dev485`, CPU, Apple Silicon)
```
ARCH: ['PhantomForCausalLM']
TOKENS: (1000, 1000, 1000, 1000)
```
The real scheduler, KV-cache manager and CPU model runner drove our model for
4 decode steps. `load_format="dummy"` means no weights are downloaded;
`skip_tokenizer_init=True` + `TokensPrompt` means no tokenizer is needed.

**Learned**
- vLLM's model contract is small: `__init__(vllm_config, prefix)`, `forward`,
  `embed_input_ids`, `compute_logits` (`interfaces_base.py`).
- Registering with `ModelRegistry` inside a script is not enough once the
  engine runs in a subprocess (macOS `spawn`): the child never runs that code.
  A `vllm.general_plugins` entry point fixes it, since vLLM loads plugins in
  each process.
- Editable installs only pick up new entry points when reinstalled
  (`uv pip install -e .` after changing `pyproject.toml`).
- Harmless warning: with `skip_tokenizer_init=True`, vLLM logs
  "Tokenizer not available"; generation is unaffected.

**Next (Day 5)**
- Write provenance tags into the real KV tensor via `get_attention_context()`
  and read them back through vLLM's block table.

## Day 5 — 2026-11-02

**Built**
- `phantomkv/tags.py`: tag layout `[MAGIC, pos, token, h0..h3, 0]`, every field
  an integer < 2^24, so it is exact in a float32 KV slot; `expected_continuation()`
  is the pure-Python reference. 6 new tests (24 total).
- `PhantomForCausalLM` now writes a tag for every new token into the slot from
  vLLM's `slot_mapping`, cross-checks it against the block table, re-reads the
  whole context through vLLM's block table, and emits the next token from the
  hash chain (`hidden_states[:, 0]` -> one-hot logits).
- `scripts/vllm_phantom_check.py`: 8 prompts with long shared prefixes,
  generated twice so round 2 hits the prefix cache.

**Result: first exact check of real vLLM KV bookkeeping, on a laptop CPU**
```
STATS: {'steps': 34, 'tokens_written': 748, 'slots_checked': 19776, 'violations': 0}
OUTPUT MISMATCHES: 0
VIOLATIONS: 0
RESULT: PASS
```
Without any cache reuse the workload would write 1,356 tags; it wrote 748.
So vLLM served 608 positions from cached blocks, and every one of the 19,776
slot reads (including through those shared blocks) was correct.

**Learned**
- vLLM computes logits only for the last token of each request
  (`logits_indices = query_start_loc[1:] - 1`), so a model can carry its
  answer in `hidden_states` and make sampling exact.
- In eager mode, warm-up/profiling runs pass no attention metadata, so the
  model can tell real steps from warm-up and never raises false alarms.
- A violation file (JSON lines) is the simplest way to get results out of a
  model that may run in another process.

**Next (Day 6)**
- Plant a bug inside vLLM (local patch) and show Phantom catches it.
- Seeded workloads + memory pressure (tiny KV cache) to force eviction and preemption.

## Day 6 — 2026-11-02

**Built**
- `phantomkv/vllm_plants.py`: known bug classes planted into real vLLM at
  runtime via `PHANTOM_VLLM_PLANT` (the plugin runs in every vLLM process, so
  the plant lands in the EngineCore). vLLM's source is never modified.
  First plant, `stale-prefix-cache`: `BlockPool._maybe_evict_cached_block`
  forgets to drop the hash -> block index entry when a cached block is evicted.
- `scripts/vllm_phantom_check.py --pressure`: 48 KV blocks, max 4 sequences,
  and a churn phase (14 unrelated prompts) between two shared-prefix phases,
  so cached blocks are evicted and reused.

**Result: Phantom catches a bug planted inside real vLLM**

| Run | vLLM | Slots checked | Violations | Wrong outputs | Result |
|---|---|---|---|---|---|
| `--pressure` | unmodified | 55,872 | 0 | 0 | PASS |
| `--pressure --plant stale-prefix-cache` | planted bug | 47,488 | **96** (`misplaced-block`) | **7 / 8** in phase C | **FAIL (caught)** |
| default (Day 5) | unmodified | 19,776 | 0 | 0 | PASS |

First violation: `pos 0: slot 32 holds position 64`. A request asked for its
prompt's first block; the stale index entry pointed it at block 1, which had
been evicted and reused for another request's positions 64-95.
Side effect visible in the stats: 224 fewer tags written (3,333 -> 3,109),
i.e. 7 blocks x 32 tokens wrongly "reused" instead of recomputed.

**Learned**
- vLLM's cache lookup (`BlockPool.get_cached_block`) trusts the index
  completely: nothing re-validates that a cached block still holds the
  prefix its hash says. Correctness rests on eviction always removing index
  entries; Phantom checks that from the outside.
- Phantom reports the same stale block again on every later step that reads
  it (96 reports for 7 bad blocks). Next: dedupe per (request, block) so
  reports stay readable.

**Next (Day 7)**
- Drive vLLM step by step from a seeded event list (`LLMEngine.add_request`,
  `abort_request`, `step`), so aborts happen at chosen moments and one seed
  fully describes a real-vLLM run.

## Day 7 — 2026-11-02

**Built**
- `phantomkv/vllm_harness.py`: seed -> event list sized for 32-token blocks
  (24 requests, shared prefixes of 1-3 whole blocks, ~25% client aborts).
  `VllmRunner` drives vLLM's real `LLMEngine` one `step()` at a time: at
  engine step k it adds/aborts exactly the events with time == k. Every
  output (including partial outputs of aborted requests) is checked against
  the reference; unfinished requests after `max_steps` = liveness failure.
  One engine reused across seeds, prefix cache reset between seeds.
- `phantomkv vllm-fuzz [--plant NAME] [--check-determinism]`.
- `python -m phantomkv` now guards `__main__` (macOS `spawn` re-imports it).

**Found and fixed a false positive in Phantom itself**
First run: seed 14 reported violations on *unmodified* vLLM, identically
with and without the plant, and two runs of the same seed differed.
Root cause: `KVCacheManager.allocate_slots()` registers a request's blocks
in the prefix cache at *schedule* time
(`cache_blocks(request, total_computed + num_new_tokens)`), so a request
scheduled later in the same step can hit blocks still being filled in that
forward pass. That is legal in vLLM because attention writes KV for the
whole batch before any read. Phantom wrote and checked one request at a
time; when the reader's batch row came first it read leftovers of the
block's previous occupant. Fix: write every new token in the batch first,
in ascending position order, then check.

**Results (after the fix)**

| Run | Seeds | Result |
|---|---|---|
| `vllm-fuzz --check-determinism` (unmodified vLLM) | 20, each run twice | **0 violations, every seed identical across runs** |
| `vllm-fuzz --plant stale-prefix-cache` | 20 | 1 seed caught (seed 12: 11 `misplaced-block`, 2 wrong outputs) |
| Day 6 `--pressure` re-run | — | unchanged: 0 / 55,872 clean; 96 violations with plant |

Throughput: ~190 requests/s on one CPU core (480 requests in 2.5 s).

**Learned**
- A bug finder's first job is to be trusted: rule out the tool before
  blaming the engine. The "two runs differ" signal pointed straight at
  leftover memory being read.
- Real vLLM on CPU replays deterministically from a seed when driven step by
  step (async scheduling off), which is what shrinking needs.
- With multiprocess engines, `step()` blocks when nothing is in flight, so
  the harness only steps while requests are live.
- Detection rate of the planted bug is low at this config (1/20 seeds):
  the stale entry is only harmful if the evicted prefix is requested again.

**Next (Day 8)**
- Shrink real-vLLM failures (seed 12) with ddmin, reusing one engine.
- Raise detection rate: workloads that re-request evicted prefixes.

## Day 8 — 2026-11-02

**Built**
- Revisit wave in `vllm_workload()`: 10 extra requests that reuse the first
  whole blocks of a random earlier prompt, arriving after the cache has
  churned. (Seeds now produce different workloads than on Day 7.)
- `shrink()` for real vLLM + `phantomkv vllm-shrink --seed N [--plant NAME]`:
  ddmin over the event list, re-running real vLLM for every candidate with
  one reused engine and the prefix cache reset each time; the minimal list is
  re-run 3 times to confirm it reproduces.

**Results**

| Run | Result |
|---|---|
| unmodified vLLM, 20 seeds x 2 runs (680 requests) | 0 violations, identical across runs |
| `--plant stale-prefix-cache`, 20 seeds | **15 / 20 caught** (Day 7: 1 / 20) |
| `vllm-shrink --seed 1 --plant stale-prefix-cache` | **38 events -> 13**, 235 vLLM runs in 12.5 s, reproduces **3/3** |

Minimal reproduction for seed 1 (same letter = same first 32-token block):
```
step 5    submit req 11  first_block=A      <- prefix A cached
step 15-79  9 requests B..H                 <- fill the 48 blocks; A's blocks evicted and reused
step 87   submit req 33  first_block=A      <- prefix A requested again
step 97   submit req 31  first_block=A      <- stale index entry hit
=> misplaced-block: slot 224 holds position 96
```

**Learned**
- Workload design matters as much as the oracle: the same planted bug went
  from 1/20 to 15/20 detection just by re-requesting evicted prefixes.
- Real-engine reproductions are larger than toy ones (13 vs ~6 events):
  eviction needs enough *other* traffic to fill the cache, and ddmin proves
  each of those requests is necessary.
- The engine is reused across ~235 runs, so violation `step` numbers are
  cumulative; next: report steps relative to the run.

**Next**
- Day 9: run the real-vLLM check in GitHub CI (Linux CPU wheels, pinned commit).

## Day 9 — 2026-11-03

**Built**
- `scripts/ci_vllm.sh`: five acceptance checks against real vLLM, in both
  directions. Unmodified vLLM must pass (prefix-cache check, memory-pressure
  check, 10 seeds each run twice); with `stale-prefix-cache` planted, the
  pressure check and the seeded fuzz must *fail*. A tool that stops catching
  the planted bug fails CI just like one that raises false alarms.
- `.github/workflows/vllm.yml` (`real-vllm`): on every push that touches code,
  installs vLLM's Linux CPU wheel on `ubuntu-latest`, preloads tcmalloc and
  Intel OpenMP (required by the CPU wheels), and runs `ci_vllm.sh`.

**Results**
- Locally (Apple Silicon): 5/5 checks OK, twice.
- GitHub Actions: green on the first push (`8c2f3b5`).

**Debugged: no wheel for the pinned commit**
`wheels.vllm.ai/0e5b2ba118.../cpu/` returned 404. The nightly index pointed at
its parent, `76c6f3f2a` (`0.33.0rc1.dev484`), which has a
`manylinux_2_39_x86_64` CPU wheel. `git diff 76c6f3f2a 0e5b2ba118` touches only
`vllm/model_executor/models/mimo_v2_omni.py`, its test and CI config, none of
which Phantom uses. So CI installs that exact wheel file (not the moving
nightly index), and the pin is documented in `docs/VLLM_INTEGRATION.md`.

**Learned**
- Pin CI to an exact artifact, not to "nightly": otherwise results change
  without any change in your code.
- When the exact version isn't available, check the diff of the nearest one
  before substituting it, and write down why it is equivalent.
- Run `git am` once per patch: a second run leaves git mid-apply
  (`git am --abort` restores the clean state).

**Phase 2 complete.** Next: Phase 3, more chaos (chunked prefill, async
scheduling, tight batch budgets, block sizes) on real vLLM.

## Day 10 — 2026-11-03

**Built (Phase 3 starts)**
- vLLM configuration profiles, each stressing a different part of KV
  bookkeeping: `baseline`, `chunked-prefill` (`max_num_batched_tokens=64`),
  `preemption` (24 blocks, 8 concurrent sequences), `block64`.
- Runs report vLLM's own counters (`num_preemptions`, `prefix_cache_hits`,
  `prefix_cache_queries`), so a clean result comes with proof that the
  stressed path was exercised.
- `vllm-fuzz --profile`, `vllm-shrink --profile`, and
  `scripts/vllm_campaign.sh` (every profile, unmodified + planted).

**Campaign (20 seeds per run, 680 requests each)**

| Profile | Unmodified vLLM | `stale-prefix-cache` planted | Engine counters (unmodified) |
|---|---|---|---|
| baseline | PASS | CAUGHT 15/20 | 0 preemptions, 33% prefix-cache hit rate |
| chunked-prefill | PASS | CAUGHT 15/20 | 0 preemptions, 33% hit rate |
| preemption | PASS | CAUGHT 20/20 | **16 preemptions**, 25% hit rate |
| block64 | PASS | CAUGHT 16/20 | 0 preemptions, 29% hit rate |

Unmodified runs use `--check-determinism` (each seed run twice).
Throughput ~100 req/s with every seed run twice, ~210-240 req/s single.

**Investigation 1: a profile that silently did nothing**
The first campaign had an `async` profile. Its counters matched baseline
*exactly* (22,688 hit tokens, 680 requests). Cause: `vllm/platforms/cpu.py`
forces `scheduler_config.async_scheduling = False` ("async scheduling is not
required on CPU"). The profile was removed; async scheduling needs a GPU run.

**Investigation 2: chunked-prefill ALARM = a timing race in the harness**
Unmodified vLLM, seed 0 "NONDETERMINISTIC" with no violations. Hypothesis 1
(free-block order after cache reset changes eviction) was *disproved*:
running seed 0 four times on one engine gave identical steps (132) and hits
(1088). The only difference: the first run on a fresh engine produced 355
output tokens, later runs 354, all correct. Cause: the default multiprocess
engine runs its own busy loop, so `abort_request` arrives asynchronously and
can land a step late; the slow first run let an aborted request emit one
more token. Fix: run the EngineCore in-process
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`). After the fix: four runs identical
(`fp=7beb8cc701bb54da`), and the campaign is all PASS / CAUGHT.

**Learned**
- Measure that a configuration actually took effect; frameworks override
  settings silently.
- Test your hypothesis before fixing: the first explanation was wrong, and
  the data said so.
- A deterministic simulator needs synchronous control of the system under
  test; message-passing between processes reintroduces timing.
- Gap: no counter directly proves chunking; next, report scheduled tokens
  per step to show prefills were split.

**Next (Day 11)**
- Prove chunking happened; add a second planted bug class so detection isn't
  tested against a single bug.

## Day 11 — 2026-11-03

**Built**
- Second planted bug class, `missing-touch`: `BlockPool.touch()` doesn't count
  a new owner when a prefix-cache hit lands on a block another running request
  still holds (refcount bug, vs. `stale-prefix-cache`'s index bug).
- Evidence of chunked prefill (vLLM has no counter for it): the model counts
  `prefill_continuations`, multi-token writes that resume exactly where the
  request's previous multi-token write ended, and the largest tokens-per-step.
- Engine exceptions are recorded with file, line and source; after one, the
  engine is marked poisoned and remaining seeds are skipped. `vllm-fuzz` exits
  1 only when *Phantom* found something, 2 when only vLLM's own check fired;
  the campaign prints CAUGHT / VLLM-ASSERT / MISSED / CRASH.

**Results**

| Profile | Unmodified | `stale-prefix-cache` | Evidence |
|---|---|---|---|
| baseline | PASS | CAUGHT 15/20 | 0 chunk continuations, up to 499 tokens/step |
| chunked-prefill | PASS | CAUGHT 15/20 | **498 chunk continuations, max 64 tokens/step** |
| preemption | PASS | CAUGHT 20/20 | 16 preemptions |
| block64 | PASS | CAUGHT 16/20 | 64-token blocks |

`missing-touch` (re-run after the harness fixes):

| Profile | What happened |
|---|---|
| baseline | seed 0: no visible corruption (freed cached blocks go to the *back* of the free queue, so the block wasn't reused while still read); seed 1, step 2: vLLM assertion |
| preemption | seed 0, step 14: vLLM assertion, before any corruption was visible |

The assertion is `vllm/v1/core/block_pool.py:730`, in
`BlockPool.get_new_blocks()`: `assert block.ref_cnt == 0` for every block taken
off the free queue. A block wrongly freed while still owned is caught when it
is handed out again.

**Finding: where Phantom adds value**
- Bugs that make vLLM's bookkeeping *inconsistent* (refcounts) are already
  caught by vLLM's own assertions.
- Bugs that leave the bookkeeping *consistent but wrong* (a valid-looking index
  entry pointing at recycled memory) trip no assertion; only an external oracle
  like Phantom sees them (`stale-prefix-cache`: 15-20 of 20 seeds in every
  profile). Phase 5's bug hunt should aim there.

**Mistakes caught today (my own)**
- The first campaign printed CAUGHT for `missing-touch` because any non-zero
  exit counted as "caught", including vLLM crashing. The logs showed
  `engine-error` on every seed after the first (a poisoned engine) and, in one
  profile, a harness `KeyError` from a request left over from the previous seed.
- Negative per-run counters came from a stale stats file of a previous process.
All fixed (see commits); `missing-touch` is not in CI, since Phantom doesn't catch it.

**Next (Day 12)**
- Start Phase 4: prefill/decode disaggregation on one laptop, with injected
  transfer faults.
