"""Phase 4: prefill/decode disaggregation on one laptop, checked by Phantom.

Two separate vLLM engines (two processes, like two machines) connected by
vLLM's ExampleConnector, which ships KV through files in a shared folder:

  1. PREFILL engine: generates 1 token per prompt; the connector saves each
     prompt's KV (Phantom's tags) to <storage>/<hash of prompt tokens>/.
  2. DECODE engine: receives prompt + that token; the connector loads the
     prompt's KV (whole blocks) into the decode engine's OWN slots; the rest
     is computed locally; decoding continues.

Phantom checks, in the decode engine, every slot of every context through
vLLM's block table, including the seam where transferred KV meets locally
computed KV, and the harness checks every output token against the reference.

    python scripts/vllm_pd_check.py            # runs both engines
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys

STORAGE = os.path.abspath("pd_storage")
PROMPTS = os.path.abspath("pd_prompts.json")
BLOCK_SIZE = 32
PAGE_BYTES = 2 * BLOCK_SIZE * 1 * 32 * 4
DECODE_TOKENS = 12


def make_prompts(seed: int = 0, n: int = 8) -> list[list[int]]:
    rng = random.Random(seed)
    shared = [rng.randrange(1, 50_000) for _ in range(64)]  # 2 blocks
    out = []
    for i in range(n):
        body = [rng.randrange(1, 50_000) for _ in range(rng.randrange(10, 140))]
        out.append(shared + body if i % 2 == 0 else body + shared[:20])
    return out


def build_llm(role: str):
    from vllm import LLM
    from vllm.config import KVTransferConfig

    return LLM(
        model="facebook/opt-125m",
        hf_overrides={"architectures": ["PhantomForCausalLM"]},
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="float32",
        enforce_eager=True,
        # Off on purpose: then any context the decode engine did not compute
        # itself can only have arrived through the connector.
        enable_prefix_caching=False,
        block_size=BLOCK_SIZE,
        max_model_len=512,
        max_num_seqs=8,
        kv_cache_memory_bytes=64 * PAGE_BYTES,
        seed=0,
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": STORAGE},
        ),
    )


def run_role(role: str) -> int:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from phantomkv.tags import expected_continuation

    report = os.path.abspath(f"phantom_report_{role}.jsonl")
    for p in (report, report + ".stats.json"):
        if os.path.exists(p):
            os.remove(p)
    os.environ["PHANTOM_REPORT"] = report
    llm = build_llm(role)
    vocab = llm.llm_engine.model_config.hf_config.vocab_size
    mismatches = 0

    if role == "prefill":
        prompts = make_prompts()
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                            SamplingParams(max_tokens=1, temperature=0, detokenize=False),
                            use_tqdm=False)
        records = []
        for p, o in zip(prompts, outs):
            t1 = list(o.outputs[0].token_ids)
            if t1 != expected_continuation(p, 1, vocab):
                mismatches += 1
            records.append({"prompt": p, "t1": t1})
        with open(PROMPTS, "w") as f:
            json.dump(records, f)
        saved = sum(len(files) for _, _, files in os.walk(STORAGE))
        print(f"PREFILL: {len(prompts)} prompts, KV files saved: {saved}")
    else:
        records = json.load(open(PROMPTS))
        prompts = [r["prompt"] + r["t1"] for r in records]
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                            SamplingParams(max_tokens=DECODE_TOKENS, temperature=0,
                                           detokenize=False),
                            use_tqdm=False)
        for p, o in zip(prompts, outs):
            got = list(o.outputs[0].token_ids)
            if got != expected_continuation(p, len(got), vocab):
                mismatches += 1
        print(f"DECODE: {len(prompts)} prompts, {DECODE_TOKENS} tokens each")

    violations = [json.loads(l) for l in open(report)] if os.path.exists(report) else []
    stats = json.load(open(report + ".stats.json")) if os.path.exists(report + ".stats.json") else {}
    print(f"{role.upper()} STATS: {stats}")
    print(f"{role.upper()} OUTPUT MISMATCHES: {mismatches}")
    print(f"{role.upper()} VIOLATIONS: {len(violations)}")
    for v in violations[:3]:
        print("  ", v)
    return 0 if (mismatches == 0 and not violations and stats) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["prefill", "decode", "both"], default="both")
    a = ap.parse_args()
    if a.role != "both":
        return run_role(a.role)

    shutil.rmtree(STORAGE, ignore_errors=True)
    os.makedirs(STORAGE)
    codes = []
    for role in ("prefill", "decode"):  # two processes, like two machines
        codes.append(subprocess.call([sys.executable, __file__, "--role", role]))
    ok = codes == [0, 0]
    print("RESULT:", "PASS" if ok else f"FAIL (exit codes {codes})")
    return 0 if ok else 1


if __name__ == "__main__":  # required on macOS (spawn re-imports this file)
    sys.exit(main())
