"""Day 5 check: Phantom tags written into vLLM's real KV cache read back intact.

Runs real vLLM (CPU) with the Phantom model on prompts that share long
prefixes, twice, so the second round hits vLLM's prefix cache. Passes iff:
  * every generated token equals the reference computed in pure Python, and
  * the model reported zero provenance violations.

    uv pip install -e .            # inside the vLLM venv, registers the plugin
    python scripts/vllm_phantom_check.py
"""

import json
import os
import random

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from phantomkv.tags import expected_continuation

REPORT = os.path.abspath("phantom_report.jsonl")


def make_prompts(seed: int = 0, n: int = 8) -> list[list[int]]:
    rng = random.Random(seed)
    shared = [rng.randrange(1, 50_000) for _ in range(80)]  # spans 2+ blocks of 32
    prompts = []
    for i in range(n):
        tail = [rng.randrange(1, 50_000) for _ in range(rng.randrange(3, 40))]
        prompts.append((shared + tail) if i % 2 == 0 else tail + shared[:20])
    return prompts


def main() -> None:
    for path in (REPORT, REPORT + ".stats.json"):
        if os.path.exists(path):
            os.remove(path)
    os.environ["PHANTOM_REPORT"] = REPORT

    llm = LLM(
        model="facebook/opt-125m",
        hf_overrides={"architectures": ["PhantomForCausalLM"]},
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="float32",
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=True,
        block_size=32,
        kv_cache_memory_bytes=64 << 20,
        seed=0,
    )
    vocab = llm.llm_engine.model_config.hf_config.vocab_size
    prompts = make_prompts()
    params = SamplingParams(max_tokens=16, temperature=0, detokenize=False)

    mismatches = 0
    for round_ in (1, 2):  # round 2 reuses round 1's cached prefixes
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
        for p, o in zip(prompts, outs):
            got = list(o.outputs[0].token_ids)
            want = expected_continuation(p, len(got), vocab)
            if got != want:
                mismatches += 1
                print(f"MISMATCH round {round_} prompt_len={len(p)}\n  got  {got}\n  want {want}")

    violations = []
    if os.path.exists(REPORT):
        violations = [json.loads(line) for line in open(REPORT)]
    stats = json.load(open(REPORT + ".stats.json")) if os.path.exists(REPORT + ".stats.json") else {}

    print("STATS:", stats)
    print("OUTPUT MISMATCHES:", mismatches)
    print("VIOLATIONS:", len(violations))
    for v in violations[:5]:
        print("  ", v)
    print("RESULT:", "PASS" if mismatches == 0 and not violations and stats else "FAIL")


if __name__ == "__main__":  # required on macOS (spawn re-imports this file)
    main()
