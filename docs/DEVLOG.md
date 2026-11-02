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
