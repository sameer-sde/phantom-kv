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

## Day 12 — 2026-11-03

**Built (Phase 4 starts)**
- Phantom's write-and-check step now runs through vLLM's own
  `maybe_transfer_kv_layer` decorator. KV connectors save and wait per layer
  from inside that decorator, which vLLM puts on its attention op; Phantom never
  calls the op, so as a prefill engine it would have saved nothing.
- `scripts/vllm_pd_check.py`: a prefill engine and a decode engine as two
  separate processes, connected by vLLM's `ExampleConnector` (KV through files
  in a shared folder). Prefix caching off in both, so any context the decode
  engine did not compute must have arrived through the connector.
- Evidence counter `context_not_computed_here`.
- `docs/PD_DESIGN.md`: how the connector moves KV, what Phantom checks, the
  fault plan for days 13-14.

**Result: KV transferred between two vLLM engines, verified exactly**
```
PREFILL: 8 prompts, KV files saved: 8
DECODE STATS: {'steps': 13, 'tokens_written': 232, 'slots_checked': 12624,
               'violations': 0, 'context_not_computed_here': 864}
DECODE OUTPUT MISMATCHES: 0
RESULT: PASS
```
The connector loads `(len(prompt) - 1) // 32 * 32` tokens per prompt: for
lengths 130, 121, 178, 151, 140, 63, 139, 78 that is 128 + 96 + 160 + 128 +
128 + 32 + 128 + 64 = **864**, exactly the context the decode engine had
without computing it. Every transferred slot, the seam to locally computed KV,
and all 96 decode tokens checked out. The decode engine wrote 232 tokens:
144 prompt tokens beyond the transferred blocks + 88 decode inputs.
Single-engine regression unchanged (55,872 slots, PASS).

**Learned**
- In vLLM, "who calls the connector" matters: loads are triggered by the
  model runner, saves by a decorator on the attention op. A model that skips
  the op must call the same hook.
- Tags carry `(pos, token, chain)`, not slot numbers, so they stay valid when
  KV moves to different slots in another engine.

**Next (Day 13)**
- `PhantomFaultConnector`: swap and drop faults as planted bugs.

## Day 13 — 2026-11-04

**Built**
- `PhantomFaultConnector` (`phantomkv/vllm_fault_connector.py`): vLLM's
  `ExampleConnector` plus planted transfer faults, loaded through
  `kv_connector_module_path` (no vLLM changes), applied on the decode side only:
  - `swap`: serve another prompt's KV file of identical size, so shapes match and
    nothing raises;
  - `silent-drop`: report the load as done, write nothing.
  Every injected fault is logged, so a run proves the fault actually fired.
- `vllm_pd_check.py --fault`, with verdicts PASS / CAUGHT / MISSED /
  NOT TRIGGERED / CRASH (a crashing engine is never counted as caught).
- The `real-vllm` CI job runs all three two-engine checks.

**Results**

| Run | Fault fired | Phantom saw | Verdict |
|---|---|---|---|
| no fault | — | 864 transferred positions, 12,624 slot reads, 0 violations, 0 mismatches | PASS |
| `swap` | 4 times | **4 wrong outputs**, 0 violations | CAUGHT |
| `silent-drop` | 2 load steps | **96 `unwritten-read`** (`slot 32 holds no tag`), 8/8 wrong outputs | CAUGHT |

Why 4 swaps: the connector only swaps files of identical size; the four
prompts that transfer 128 tokens rotated among themselves, the other four
had no same-size partner. 4 swaps -> exactly 4 wrong outputs.

**Finding: why Phantom has two independent oracles**
- `swap` delivers a *valid chain for the wrong prompt*. Inside the model every
  slot correctly extends the one before it, so the in-model checks pass. Only
  the harness's reference check (outputs vs. the true prompt) catches it.
- `silent-drop` leaves slots empty; the in-model check catches it on the first read.
Each oracle catches a class the other cannot; this was predicted in the
Day 3 design and is now demonstrated on real vLLM across two engines.

**Next (Day 14)**
- Delay + abort: a transfer that lands after its request was aborted and its
  blocks reused (the SGLang Oct-2026 bug class).

## Day 14 — 2026-11-05

**Built**
- `PhantomAsyncFaultConnector`: every external hit is an *asynchronous* load,
  so vLLM parks the request in `WAITING_FOR_REMOTE_KVS` with blocks allocated;
  the KV lands 3 engine steps later, one-sided like an RDMA read (written into
  the blocks given at start, regardless of current owner), and completion is
  reported through `get_finished()`.
- Third planted bug, `abort-free-race` (the SGLang Oct-2026 class): the
  scheduler frees an aborted request's blocks immediately even though its load
  is in flight, and silently drops the late completion.
- `vllm_pd_check.py --abort-mid-transfer [--plant abort-free-race]`: decode
  engine stepped in-process; 8 transferred requests start loading, 4 are
  aborted while their KV is in flight, then 8 filler requests take whatever
  blocks are free. Both runs added to CI.

**vLLM's defence (read before building)**
`Scheduler.finish_requests()` passes `delay_free_blocks=True` for a request in
`WAITING_FOR_REMOTE_KVS` (unless its load never started or already finished),
and `_update_from_kv_xfer_finished()` frees the blocks only when the worker
reports the load done.

**Results**

| Run | Aborted in flight | Late landings | Phantom saw | Verdict |
|---|---|---|---|---|
| unmodified vLLM | 4 (`d0 d2 d4 d6`) | 4 | 14,220 slot reads, 0 violations, 0 wrong outputs | PASS |
| `abort-free-race` planted | 4 | 4 | **54 `misplaced-block`** (all at step 4, when the late loads landed), **5 fillers with wrong outputs** | CAUGHT |

First violation with the plant: `slot 800 holds position 96`. A filler's
position 0 was overwritten by an aborted request's position 96. Unmodified,
the aborted requests' blocks were held until their KV landed, and the fillers
got clean blocks. `context_not_computed_here: 320` = the 4 completed transfers
(lengths 121, 151, 63, 78 -> 96 + 128 + 32 + 64).

**Learned**
- vLLM already defends against this bug class; Phantom verifies the defence
  end to end and shows exactly what breaks without it.
- An async connector must write through the KV tensors from
  `register_kv_caches()`: steps without a forward pass have no attention
  metadata.
- Patch files can be written straight into the repo's parent folder over the
  desktop link, instead of downloading from the chat.

**Next (Day 15)**
- Seeded PD fuzzing: random abort timing, delays and fillers across many seeds,
  plus shrinking, so the two-engine path gets the same treatment as the
  single-engine one.

## Day 15 — 2026-11-05

**Built**
- `phantomkv/vllm_pd.py`: seeded fuzzing of disaggregated prefill/decode on
  real vLLM. A pool of 16 prompts is prefilled once by a separate engine and
  reused. Each seed picks the async load delay (1-5 engine steps), which pool
  prompts to transfer and when, which to abort (before the load starts, while
  KV is in flight, after it lands) and when local fillers arrive to take freed
  blocks. One in-process decode engine runs seed after seed and never saves KV,
  so a replay of a seed sees exactly the same remote hits.
- Every abort is classified from the connector's start/landing log, so a
  campaign reports how many aborts really happened while KV was in flight.
- `phantomkv pd-fuzz` and `phantomkv pd-shrink` (ddmin over the events, load
  delay held fixed). Both PD fuzz runs added to the `real-vllm` CI job.
- Day 14's async connector: delay read per load, load starts logged, swap and
  silent-drop faults also apply to async loads.
- First pull-request day: work lands on a branch through PR #2; releases
  v0.1.0-v0.4.0 published for the finished phases (PR #1).

**Results** (20 seeds, same scenarios in both runs)

| Run | Seeds failing | Evidence |
|---|---|---|
| unmodified vLLM, every seed run twice | **0/20**, all deterministic | 137 async loads started and landed, 94 aborts, **35 while KV was in flight** |
| `abort-free-race` planted | **14/20 caught** | 93 `misplaced-block`, 18 `chain-break`, 19 wrong outputs |

The 6 seeds the plant did not break: seeds 2, 3 and 8 had no abort while KV
was in flight (the bug cannot fire); in seeds 1, 4 and 11 the freed blocks
were not reused before the late KV landed (the bug fired harmlessly). No
seed without an in-flight abort was flagged.

**Shrinking** seed 0 (delay 4): 24 events -> **2** in 36 vLLM runs (0.4 s),
reproduces 3/3:

```
step 3  submit req 1  prompt_len=130  TRANSFERRED (KV from prefill engine)
step 4  abort  req 1
-> misplaced-block: slot 416 holds position 96   (the next request's position 0)
```

One transferred request, aborted one step after its load started. The next
request (here the harness's keep-alive) is given the freed blocks, and the
late KV lands on top of it. That is the whole bug in two events.

**Mistakes I made and fixed**
- Quieting vLLM's connector logs by setting the logger level did nothing:
  vLLM's start-up runs `logging.config.dictConfig`, which resets child
  loggers to `NOTSET`. A logger filter survives that (checked with
  `dictConfig` directly).
- The transferred-token count read 480 for 20 seeds, far too low. The model's
  `context_not_computed_here` stat keys requests by their first block id,
  which a long-lived engine reuses across seeds. The PD runner now counts
  tokens from the connector's landing log instead.

**Next (Day 16)**
- Longer PD campaigns (hundreds of seeds), the swap and silent-drop faults
  under fuzzing, and the Phase 4 wrap-up (release v0.5.0).

## Day 16 — 2026-11-05

**Built**
- Fourth transfer fault, `duplicate`: an async load lands correctly, then a
  stale copy of the same write lands 6 steps later without vLLM knowing, like
  a retried RDMA write. Harmless while the request still owns its blocks;
  corrupting once they are freed and reused.
- `pd-fuzz` keeps stepping until stale copies have landed (nothing leaks into
  the next seed) and counts every injected fault per run; it warns if a fault
  never fired.
- `scripts/pd_campaign.sh`: one decode engine per mode, 200 seeds each, one
  verdict line per run, and the seeds that were not caught named.
- `duplicate` added to the `real-vllm` CI job.

**Campaign** (200 seeds per run, identical scenarios in every run)

| Decode engine | Result | Failing seeds |
|---|---|---|
| unmodified vLLM (each seed run twice) | PASS, deterministic | **0/200** |
| `abort-free-race` planted | CAUGHT | 115/200 |
| `swap` fault | CAUGHT | 199/200 |
| `silent-drop` fault | CAUGHT | 199/200 |
| `duplicate` fault | CAUGHT | 184/200 |

Evidence, the same in every run: 1,371 async loads started and landed, 995
aborts, **335 while KV was in flight**, 73,376 transferred tokens read by
requests that survived. Every planted fault fired on every load (1,371).

**The one seed swap and silent-drop missed.** Both missed seed 159. My guess
was that the wrong KV happened to equal the right KV (two pool prompts share
their first two blocks). Wrong: the per-seed log shows
`transferred_tokens=0`. In seed 159 every request whose KV was transferred
was aborted before it read that KV (and the one survivor's prompt is shorter
than a block, so nothing was transferred for it). No request read transferred
data, so a broken transfer has nothing to corrupt. Missing it is correct,
and the evidence counter is what shows that.

**Shrinking the new fault:** seed 0, 24 events -> 3, reproduces 3/3:

```
step 3  submit req 1  TRANSFERRED      step 4  abort req 1
step 8  submit req 3  TRANSFERRED
-> misplaced-block at req 3's position 64: slot 512 holds position 96
```

vLLM correctly holds req 1's blocks until its load lands, then frees them;
req 3 is given them, and the stale copy of req 1's write lands on top.

**Phase 4 done.** Two real vLLM engines, KV transferred between them,
four planted faults and a planted scheduler bug, all caught and in CI; seeded
fuzzing and shrinking across both engines. Release v0.5.0.

**Not done from the original Phase 4 plan:** reorder and handshake-failure
faults (the file-based connector has no ordering or handshake to break), and
the NIXL metadata bug class, which needs the NIXL connector. Moved to Phase 5.

**Next (Phase 5)**
- Point Phantom at code nobody planted bugs in: newer vLLM KV paths (CPU
  offloading, hybrid KV groups, deferred frees), then SGLang and LMCache.

## Day 17 — 2026-11-05

**Phase 5 starts: vLLM code nobody planted bugs in.** Target 1 is vLLM's
`SimpleCPUOffloadConnector`, whose scheduler-side manager (~1,500 lines)
decides which GPU blocks are copied to a CPU cache and which CPU blocks a
request may load back. Design: [OFFLOAD_DESIGN.md](OFFLOAD_DESIGN.md).

**Built**
- The connector and its manager run unmodified on the CPU backend; only the
  CUDA/DMA copy engine is replaced (`PhantomOffloadWorker`, installed by the
  plugin), executing the same block lists with tensor copies.
- Six `vllm-fuzz` profiles: `cpu-offload`, `-lazy`, `-slow` (2-step
  transfers), `-small` (32 GPU + 32 CPU blocks), `-very-slow` (25-step
  transfers), `-preempt` (8 sequences, 24 GPU blocks).
- Evidence per campaign: blocks stored, blocks loaded back, flushes, resets
  refused.

**Results, unmodified vLLM** (10 seeds each, every seed run twice)

| Profile | Result | Stored / loaded back | Other |
|---|---|---|---|
| cpu-offload | clean, deterministic | 616 / 31 | |
| cpu-offload-lazy | clean, deterministic | 559 / 27 | |
| cpu-offload-slow | clean, deterministic | 616 / 32 | |
| cpu-offload-small | clean, deterministic | 696 / 14 | |
| cpu-offload-preempt | clean, deterministic | 634 / 146 | 6 preemptions, 6 flushes |
| cpu-offload-very-slow | **no violations, but 9/10 seeds differ on replay** | 632 / 67 | open lead |

No Phantom violation and no wrong output anywhere: 3,753 blocks stored,
317 loaded back and verified.

**Mistakes and dead ends**
- First run: every seed "nondeterministic". The harness reset only the GPU
  prefix cache; `reset_prefix_cache(reset_connector=True)` also resets the
  CPU cache. Fixed. The CLI then blamed "vLLM raised its own exception",
  which was wrong; it now reports replay differences as such.
- `cpu-offload-small` with 24 CPU blocks never loaded anything back: blocks
  were evicted before the revisit wave. Retuned.
- Planted fault 1, `store-before-compute` (the race vLLM's worker guards
  against, #45704): fired 256 times, changed nothing. The manager only stores
  blocks that were full before the step, and the CPU backend runs the
  forward pass synchronously; that race needs overlapped GPU execution.
- Planted fault 2, `store-reported-early`: fired 256 (2-step) and 283
  (25-step) times, still nothing caught. Reason found: my copy engine runs
  loads and stores in **one** FIFO queue, so a load can never overtake the
  store it depends on. vLLM's real worker uses **separate** load and store
  streams, where it can. My copy engine was less adversarial than the
  hardware it replaces.

**Open lead: cpu-offload-very-slow replays differ.** Outputs stay correct
(no mismatch against the reference), but a seed's first run takes ~70 more
steps than its replay (e.g. 225 vs 157), on 9 of 10 seeds, with every reset
accepted. The first run of seed k follows a different seed; its replay
follows itself. Either state survives `reset_prefix_cache(reset_connector=True)`
when transfers are slow, or my copy engine carries something across runs.
Not a finding until that is settled.

**Next (Day 18)**
- Separate load and store queues in the copy engine (like the real streams),
  then re-validate the planted fault.
- Trace both runs of a very-slow seed step by step and find the first
  divergence.

## Day 18 — 2026-11-05

Both Day 17 open items are closed. Neither was a vLLM bug; both were in my
harness, and the tools built to find them stay useful.

**Built**
- The offload copy engine now has **separate load and store queues**, each
  FIFO with its own delay, like vLLM's two CUDA streams. A load can now
  complete before an earlier, slower store, which the single queue made
  impossible. New profile `cpu-offload-slow-stores`: stores 40 steps,
  loads 1.
- Per-step run traces in the harness, and **`phantomkv vllm-diff`**: run a
  seed twice and print the first step where the two runs differ.

**Item 1: the planted offload fault is caught.** With separate queues,
`store-reported-early` breaks 4/10 seeds (132 `misplaced-block`, 5 wrong
outputs), e.g. `slot 1024 holds position 160`: a request loaded a CPU block
whose store had been reported done but had not run yet. Unmodified vLLM,
same profile: 0 violations.

**Item 2: the replay difference was the harness.** `vllm-diff` showed it in
the first trace line:

```
first run:  start [reset_prefix_cache -> True]
replay:     start [reset_prefix_cache -> False]
```

Every replay started on the previous run's caches. vLLM refused the reset
correctly: the offload manager still held blocks for stores it had not
issued yet. vLLM's own engine loop keeps stepping while
`scheduler.has_requests()` is true, which includes a connector's pending
push work; my harness stopped as soon as no request was live, so those
stores never went out. The harness now steps while `has_requests()` is
true. My "resets refused" counter only looked at first runs, so it showed 0
and hid this; it now counts every run.

**After the fix** (10 seeds each, every seed run twice)

| Profile | Result | Stored / loaded back |
|---|---|---|
| cpu-offload-very-slow | clean, deterministic (seed 0: identical step by step) | 614 / 67 |
| cpu-offload-slow-stores | clean, deterministic | 612 / 61, 3 flushes |
| cpu-offload | clean, deterministic | 616 / 31 |
| baseline (no offload, regression) | clean, deterministic | - |
| `store-reported-early` planted | **caught 4/10** | |

CPU offloading (slow-stores, clean and planted) added to the `real-vllm` CI job.

**Learned**
- Two "mysteries" in a row were my own harness. A diff tool that shows the
  first divergent step turned guessing into one line of evidence.
- A substitute component has to be at least as adversarial as the real one
  (two queues, not one), or the test proves less than it seems to.

**Status of target 1:** vLLM's CPU offloading manager is clean under every
profile tried, including slow stores, very slow transfers, preemption and
small caches, with the harness now validated in both directions. No bug
found here so far.

## Day 19 — 2026-11-06

**Phase 5, target 2: vLLM's `OffloadingConnector`.** vLLM's second, larger
offloading system: a ~2,600-line scheduler plus `CPUOffloadingManager`,
with chunks of several GPU blocks, synchronous loads, deferred stores, and
transfer failures as part of its API. Design:
[OFFLOAD2_DESIGN.md](OFFLOAD2_DESIGN.md).

**Built**
- `PhantomCPUOffloadingSpec`: vLLM's own `CPUOffloadingSpec` with one
  override, `get_worker()`, returning a tensor-copy worker. Selected through
  vLLM's out-of-tree spec mechanism, so **no vLLM code is patched** this time.
- `phantomkv/offload_map.py`: the block-to-chunk mapping (including a partial
  first chunk), pure Python, 4 unit tests.
- Profiles `offload2`, `-chunk2`, `-slow-stores`, `-preempt`, `-small`,
  `-sync`; faults `store-reported-early` (planted), `load-fails` and
  `store-fails` (storage failures vLLM must recover from).

**Results** (10 seeds each; clean profiles run every seed twice)

| Run | Result | Evidence |
|---|---|---|
| offload2 | clean, deterministic | 515 blocks stored, 30 loaded back |
| offload2-chunk2 | clean, deterministic | 486 stored, 12 loaded back (2 blocks per chunk) |
| offload2-slow-stores | clean, deterministic | 42 flush waits |
| offload2-preempt | clean, deterministic | 6 preemptions, 138 loaded back |
| offload2-small | clean, deterministic | 568 stored, 34 loaded back |
| `store-reported-early` planted | **caught 4/10** | 132 `misplaced-block`, 5 wrong outputs |
| every 3rd **load fails** | **outputs correct** | 6 loads failed; vLLM recomputed |
| every 3rd **store fails** | **outputs correct** | 70 stores failed; none served |

Everything ran on the first attempt. The two failure runs matter most: they
are failures real storage produces, not bugs we planted, and vLLM handled
both correctly.

**Gaps found in my own coverage**
- `sync_loads=0` in every profile: vLLM only issues synchronous loads when
  `sync_load_budget_bytes` is set. Added `offload2-sync` (results next day).
- `offload2-chunk2` loaded only 12 blocks back: partial-chunk loads are
  barely exercised. Needs a workload that revisits prefixes at odd block
  offsets.
- `pytest` is not installed in my Mac's vLLM venv (CI runs the unit tests).

OffloadingConnector added to the `real-vllm` CI job: preemption with the
determinism check, failed loads, failed stores.

**Next (Day 20):** sync loads; a revisit workload for partial chunks.
