"""End-to-end check of real vLLM's KV-cache bookkeeping with Phantom.

    uv pip install -e .                                   # inside the vLLM venv
    python scripts/vllm_phantom_check.py                  # Day 5: prefix-cache reuse
    python scripts/vllm_phantom_check.py --pressure       # tiny cache: eviction + reuse
    python scripts/vllm_phantom_check.py --pressure --plant stale-prefix-cache

Phases (all generated with greedy sampling, compared token by token with the
pure-Python reference):
  A  prompts sharing long prefixes            (fills the prefix cache)
  B  unique "churn" prompts   [--pressure]    (forces eviction: blocks are reused)
  C  phase A again                            (prefix-cache hits on whatever survived)

PASS iff zero output mismatches and zero provenance violations.
"""

import argparse
import json
import os
import random
import sys

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from phantomkv.tags import expected_continuation

REPORT = os.path.abspath("phantom_report.jsonl")
BLOCK_SIZE = 32
# Phantom's KV page: 2 (K,V) x 32 tokens x 1 head x 32 dims x 4 bytes = 8 KiB.
PAGE_BYTES = 2 * BLOCK_SIZE * 1 * 32 * 4


def shared_prefix_prompts(rng: random.Random, n: int = 8) -> list[list[int]]:
    shared = [rng.randrange(1, 50_000) for _ in range(80)]  # 2.5 blocks
    out = []
    for i in range(n):
        tail = [rng.randrange(1, 50_000) for _ in range(rng.randrange(3, 40))]
        out.append((shared + tail) if i % 2 == 0 else tail + shared[:20])
    return out


def churn_prompts(rng: random.Random, n: int = 14) -> list[list[int]]:
    return [[rng.randrange(1, 50_000) for _ in range(rng.randrange(120, 180))]
            for _ in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pressure", action="store_true",
                    help="tiny KV cache + churn phase to force block eviction and reuse")
    ap.add_argument("--plant", default=None, help="plant a known bug into vLLM")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    for path in (REPORT, REPORT + ".stats.json"):
        if os.path.exists(path):
            os.remove(path)
    os.environ["PHANTOM_REPORT"] = REPORT
    if a.plant:
        os.environ["PHANTOM_VLLM_PLANT"] = a.plant  # inherited by EngineCore

    num_blocks = 48 if a.pressure else 8192
    llm = LLM(
        model="facebook/opt-125m",
        hf_overrides={"architectures": ["PhantomForCausalLM"]},
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="float32",
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
        max_model_len=512,
        max_num_seqs=4 if a.pressure else 16,
        kv_cache_memory_bytes=num_blocks * PAGE_BYTES,
        seed=0,
    )
    vocab = llm.llm_engine.model_config.hf_config.vocab_size
    rng = random.Random(a.seed)
    phase_a = shared_prefix_prompts(rng)
    phases = [("A", phase_a)]
    if a.pressure:
        phases.append(("B", churn_prompts(rng)))
    phases.append(("C", phase_a))

    params = SamplingParams(max_tokens=16, temperature=0, detokenize=False)
    mismatches = 0
    for name, prompts in phases:
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params,
                            use_tqdm=False)
        for p, o in zip(prompts, outs):
            got = list(o.outputs[0].token_ids)
            if got != expected_continuation(p, len(got), vocab):
                mismatches += 1
        print(f"PHASE {name}: {len(prompts)} prompts done")

    violations = [json.loads(l) for l in open(REPORT)] if os.path.exists(REPORT) else []
    stats_path = REPORT + ".stats.json"
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    kinds: dict[str, int] = {}
    for v in violations:
        kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1

    print(f"CONFIG: pressure={a.pressure} kv_blocks={num_blocks} plant={a.plant or 'none'}")
    print("STATS:", stats)
    print("OUTPUT MISMATCHES:", mismatches)
    print("VIOLATIONS:", len(violations), kinds)
    for v in violations[:3]:
        print("  ", v)
    ok = mismatches == 0 and not violations and bool(stats)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":  # required on macOS (spawn re-imports this file)
    sys.exit(main())
