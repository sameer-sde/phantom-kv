"""Smoke test: does vLLM (CPU build) run end to end on this machine?

    python scripts/vllm_smoke.py

Expected last lines:
    vLLM 0.33.0rc1.dev485+g0e5b2ba11
    OUTPUT: '...'
"""

import vllm
from vllm import LLM, SamplingParams


def main() -> None:
    print("vLLM", vllm.__version__)
    llm = LLM(
        model="facebook/opt-125m",
        dtype="float32",
        enable_prefix_caching=True,
        kv_cache_memory_bytes=1 << 30,  # 1 GiB; the default reserves ~90% of RAM
    )
    out = llm.generate(["The capital of France is"],
                       SamplingParams(max_tokens=8, temperature=0))
    print("OUTPUT:", repr(out[0].outputs[0].text))


if __name__ == "__main__":  # required on macOS (spawn re-imports this file)
    main()
