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

    python scripts/vllm_pd_check.py                      # no fault: must PASS
    python scripts/vllm_pd_check.py --fault swap         # must be CAUGHT
    python scripts/vllm_pd_check.py --fault silent-drop  # must be CAUGHT
    python scripts/vllm_pd_check.py --abort-mid-transfer                          # must PASS
    python scripts/vllm_pd_check.py --abort-mid-transfer --plant abort-free-race  # must be CAUGHT

--abort-mid-transfer: decode loads are asynchronous (they land 3 steps after
starting); half the decode requests are aborted while their KV is in flight,
then "filler" requests arrive and take whatever blocks are free. Correct vLLM
keeps an aborted request's blocks until its load lands; with abort-free-race
planted, the late write lands in a filler's blocks.

Faults are injected by PhantomFaultConnector (phantomkv/vllm_fault_connector.py)
on the decode side only.
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
FAULT_LOG = os.path.abspath("pd_faults.jsonl")
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
        kv_cache_memory_bytes=96 * PAGE_BYTES,
        seed=0,
        kv_transfer_config=KVTransferConfig(
            # ExampleConnector + optional planted faults (PHANTOM_PD_FAULT);
            # the async variant when PHANTOM_PD_ASYNC_DELAY is set (decode side)
            kv_connector=("PhantomAsyncFaultConnector" if os.environ.get("PHANTOM_PD_ASYNC_DELAY")
                          else "PhantomFaultConnector"),
            kv_connector_module_path="phantomkv.vllm_fault_connector",
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
    elif os.environ.get("PHANTOM_PD_ABORT"):
        mismatches = decode_with_aborts(llm, vocab)
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


def decode_with_aborts(llm, vocab: int) -> int:
    """Step-by-step decode: abort half the requests while their KV is in flight,
    then let filler requests take the free blocks. Returns output mismatches."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from phantomkv.tags import expected_continuation

    engine = llm.llm_engine
    records = json.load(open(PROMPTS))
    prompts = {f"d{i}": r["prompt"] + r["t1"] for i, r in enumerate(records)}
    rng = random.Random(1)
    fillers = {f"f{j}": [rng.randrange(1, 50_000) for _ in range(rng.randrange(60, 120))]
               for j in range(8)}  # random: no saved KV, computed locally
    params = SamplingParams(max_tokens=DECODE_TOKENS, temperature=0, detokenize=False)

    for rid, p in prompts.items():
        engine.add_request(rid, TokensPrompt(prompt_token_ids=p), params)
    engine.step()  # admitted: blocks allocated, async loads start (land 3 steps later)
    aborted = [rid for i, rid in enumerate(prompts) if i % 2 == 0]
    engine.abort_request(aborted)  # KV for these is still in flight
    print(f"ABORTED while KV in flight: {aborted}")
    for rid, p in fillers.items():  # arrive right after: take whatever blocks are free
        engine.add_request(rid, TokensPrompt(prompt_token_ids=p), params)

    outputs: dict[str, list[int]] = {}
    for _ in range(500):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[str(out.request_id).split("-")[0]] = list(out.outputs[0].token_ids)

    all_prompts = {**prompts, **fillers}
    mismatches = 0
    for rid, got in outputs.items():
        if rid in aborted:
            continue
        if got != expected_continuation(all_prompts[rid], len(got), vocab):
            mismatches += 1
            print(f"  wrong output: {rid}")
    landed_after_abort = 0
    if os.path.exists(FAULT_LOG):
        for line in open(FAULT_LOG):
            ev = json.loads(line)
            if ev.get("event") == "landed" and str(ev.get("req", "")).split("-")[0] in aborted:
                landed_after_abort += 1
    print(f"DECODE: {len(outputs) - len(set(outputs) & set(aborted))} requests completed "
          f"({len(prompts) - len(aborted)} transferred + {len(fillers)} fillers); "
          f"late landings for aborted requests: {landed_after_abort}")
    return mismatches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["prefill", "decode", "both"], default="both")
    ap.add_argument("--fault", choices=["swap", "silent-drop"], default=None,
                    help="plant a transfer fault on the decode side")
    ap.add_argument("--abort-mid-transfer", action="store_true",
                    help="async decode loads; abort half the requests while KV is in flight")
    ap.add_argument("--plant", default=None,
                    help="plant a vLLM scheduler bug in the decode engine (e.g. abort-free-race)")
    a = ap.parse_args()
    if a.role != "both":
        try:
            return run_role(a.role)  # 0: clean, 1: Phantom found violations/mismatches
        except Exception as e:  # engine crashed: never count that as "caught"
            print(f"{a.role.upper()} CRASHED: {type(e).__name__}: {e}"[:400])
            return 3

    shutil.rmtree(STORAGE, ignore_errors=True)
    os.makedirs(STORAGE)
    if os.path.exists(FAULT_LOG):
        os.remove(FAULT_LOG)
    codes = []
    for role in ("prefill", "decode"):  # two processes, like two machines
        env = dict(os.environ)
        env.pop("PHANTOM_PD_FAULT", None)
        env.pop("PHANTOM_VLLM_PLANT", None)
        if role == "decode" and a.fault:
            env["PHANTOM_PD_FAULT"] = a.fault
            env["PHANTOM_PD_FAULT_LOG"] = FAULT_LOG
        if role == "decode" and a.abort_mid_transfer:
            env.update(PHANTOM_PD_ABORT="1", PHANTOM_PD_ASYNC_DELAY="3",
                       PHANTOM_PD_FAULT_LOG=FAULT_LOG,
                       VLLM_ENABLE_V1_MULTIPROCESSING="0")  # exact step ordering
            if a.plant:
                env["PHANTOM_VLLM_PLANT"] = a.plant
        codes.append(subprocess.call([sys.executable, __file__, "--role", role], env=env))

    if a.abort_mid_transfer:
        landed = [json.loads(l) for l in open(FAULT_LOG)] if os.path.exists(FAULT_LOG) else []
        if codes[0] != 0:
            print(f"RESULT: FAIL (prefill engine failed, exit {codes[0]})")
            return 1
        if not landed:
            print("RESULT: NOT TRIGGERED (no async load ever landed)")
            return 1
        if not a.plant:
            ok = codes[1] == 0
            print("RESULT:", "PASS" if ok else
                  {1: "ALARM (violations on unmodified vLLM: investigate)",
                   3: "CRASH (decode engine raised)"}.get(codes[1], f"FAIL (exit {codes[1]})"))
            return 0 if ok else 1
        verdict = {1: "CAUGHT", 0: "MISSED (decode looked clean)",
                   3: "CRASH (decode engine raised)"}
        print("RESULT:", verdict.get(codes[1], f"UNKNOWN (decode exit {codes[1]})"))
        return 0 if codes[1] == 1 else 1

    injected = sum(1 for _ in open(FAULT_LOG)) if os.path.exists(FAULT_LOG) else 0
    if not a.fault:
        ok = codes == [0, 0]
        print("RESULT:", "PASS" if ok else f"FAIL (exit codes {codes})")
        return 0 if ok else 1
    print(f"FAULT: {a.fault}, injected {injected} time(s)")
    if injected == 0:
        print("RESULT: NOT TRIGGERED (the fault never fired; nothing was tested)")
        return 1
    if codes[0] != 0:
        print(f"RESULT: FAIL (prefill engine failed, exit {codes[0]})")
        return 1
    caught = codes[1] == 1
    verdict = {1: "CAUGHT", 0: "MISSED (decode looked clean)", 3: "CRASH (decode engine raised)"}
    print("RESULT:", verdict.get(codes[1], f"UNKNOWN (decode exit {codes[1]})"))
    return 0 if caught else 1


if __name__ == "__main__":  # required on macOS (spawn re-imports this file)
    sys.exit(main())
