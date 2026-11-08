"""Phase 5, target 3 smoke test: Phantom inside vLLM with LMCache as the KV store.

Needs an LMCache server already running (CPU-only mode), e.g.:
    lmcache server --l1-size-gb 1 --chunk-size 32 --port 5555 --eviction-policy LRU

Pass A computes N prompts; vLLM's LMCacheMPConnector stores their KV in the
LMCache server. The GPU prefix cache is then reset, so in pass B the same
prefixes can only come back from LMCache. Phantom checks every slot read in
both passes and every output token against the pure-Python reference, and
the model's write counter shows how much of pass B was loaded rather than
recomputed.

    python scripts/lmcache_smoke.py [--port 5555] [--prompts 8]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--prompts", type=int, default=8)
    a = ap.parse_args()

    report = os.path.abspath("phantom_lmcache_report.jsonl")
    for p in (report, report + ".stats.json"):
        if os.path.exists(p):
            os.remove(p)
    os.environ.update(PHANTOM_REPORT=report, VLLM_ENABLE_V1_MULTIPROCESSING="0")

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.inputs import TokensPrompt

    from phantomkv.tags import expected_continuation
    from phantomkv.vllm_harness import page_bytes

    llm = LLM(
        model="facebook/opt-125m",
        hf_overrides={"architectures": ["PhantomForCausalLM"]},
        load_format="dummy", skip_tokenizer_init=True, dtype="float32",
        enforce_eager=True, enable_prefix_caching=True, block_size=32,
        max_model_len=512, max_num_seqs=4, kv_cache_memory_bytes=64 * page_bytes(32),
        seed=0,
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_connector_module_path="lmcache.integration.vllm.lmcache_mp_connector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": a.port,
                "lmcache.mp.mp_transfer_mode": "engine_driven",  # CPU: worker-side copies
            },
        ),
    )
    vocab = llm.llm_engine.model_config.hf_config.vocab_size
    rng = random.Random(0)
    prompts = [[rng.randrange(1, 50_000) for _ in range(rng.randrange(96, 200))]
               for _ in range(a.prompts)]
    params = SamplingParams(max_tokens=4, temperature=0, detokenize=False)

    def stats() -> dict:
        try:
            return json.load(open(report + ".stats.json"))
        except (OSError, ValueError):
            return {}

    def run(label: str, batch: list[list[int]]) -> int:
        before = stats().get("tokens_written", 0)
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in batch], params,
                            use_tqdm=False)
        bad = sum(list(o.outputs[0].token_ids) != expected_continuation(p, 4, vocab)
                  for p, o in zip(batch, outs))
        written = stats().get("tokens_written", 0) - before
        print(f"{label}: {len(batch)} prompts, {sum(map(len, batch))} prompt tokens, "
              f"tokens computed here={written}, wrong outputs={bad}")
        return bad

    bad = run("pass A (compute, LMCache stores)", prompts)
    ok = llm.reset_prefix_cache()
    print(f"GPU prefix cache reset: {ok}")
    # same prefixes, new suffix: anything not recomputed came from LMCache
    second = [p + [rng.randrange(1, 50_000) for _ in range(5)] for p in prompts]
    bad += run("pass B (prefixes must come from LMCache)", second)

    violations = [json.loads(l) for l in open(report)] if os.path.exists(report) else []
    print(f"Phantom violations: {len(violations)}")
    for v in violations[:5]:
        print("  ", v)
    s = stats()
    print(f"model stats: steps={s.get('steps')} slots_checked={s.get('slots_checked')} "
          f"context_not_computed_here={s.get('context_not_computed_here')}")
    verdict = "PASS" if not (bad or violations) else "CAUGHT"
    print(f"RESULT: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":  # required: vLLM may spawn worker processes
    sys.exit(main())
