"""Day 4 check: does real vLLM load PhantomForCausalLM and run it?

Prerequisite (inside the vLLM venv):
    uv pip install -e /path/to/phantom-kv        # registers the vLLM plugin

Run:
    python scripts/vllm_phantom_hello.py

Expected last lines:
    ARCH: ['PhantomForCausalLM']
    TOKENS: (1000, 1000, 1000, 1000)
"""

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main() -> None:
    llm = LLM(
        model="facebook/opt-125m",  # only its config is used: vocab size, hidden size
        hf_overrides={"architectures": ["PhantomForCausalLM"]},
        load_format="dummy",  # no weights to download or load
        skip_tokenizer_init=True,  # prompts are token ids
        dtype="float32",  # tags must be exact integers in the KV cache
        enforce_eager=True,  # the model has Python side effects: no graphs/compile
        enable_prefix_caching=True,
        block_size=32,  # smallest block size the CPU kernel allows
        kv_cache_memory_bytes=64 << 20,  # small on purpose
        seed=0,
    )
    print("ARCH:", llm.llm_engine.model_config.architectures)
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=[11, 22, 33, 44, 55])],
        SamplingParams(max_tokens=4, temperature=0, detokenize=False),
    )
    print("TOKENS:", tuple(out[0].outputs[0].token_ids))


if __name__ == "__main__":  # required on macOS (spawn re-imports this file)
    main()
