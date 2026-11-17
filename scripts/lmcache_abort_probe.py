"""Which abort leaves LMCache read locks behind? (Day 24)

One engine (profile lmcache-only: vLLM's own prefix cache off, so every reuse
comes from LMCache). For each case: request A stores a 128-token prefix in
LMCache; request B shares that prefix, so B's lookup hits and LMCache
read-locks the 4 chunks for it; B is then aborted at one chosen moment (or
runs to completion, as a control). Once the engine is idle no request is left,
so LMCache should hold no read locks: anything above the count before B is a
lock nobody will release (until LMCache's read TTL, 300 s by default).

Run in the vLLM + LMCache env:  python scripts/lmcache_abort_probe.py
"""

from __future__ import annotations

import random
import sys

from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from phantomkv.vllm_harness import VllmRunner, profile_config

PREFIX = 128  # 4 LMCache chunks of 32 tokens
REPEATS = 3

runner = VllmRunner(profile_config("lmcache-only"))
engine = runner.engine
rng = random.Random(0)
_n = 0


def new_id() -> str:
    global _n
    _n += 1
    return f"p{_n}"


def tokens(n: int) -> list[int]:
    return [rng.randrange(1, 50_000) for _ in range(n)]


def add(rid: str, prompt: list[int], max_new: int) -> None:
    engine.add_request(rid, TokensPrompt(prompt_token_ids=prompt),
                       SamplingParams(max_tokens=max_new, temperature=0, detokenize=False))


def step() -> None:
    runner._settle_lmcache()
    engine.step()


def drain(limit: int = 1000) -> None:
    for _ in range(limit):
        if not runner._engine_busy():
            return
        step()
    sys.exit("engine did not go idle")


def read_locked() -> int:
    st = runner.lmcache.l1_status()
    if st is None:
        sys.exit("LMCache /status unavailable")
    return st["read_locked"]


def case(name: str) -> tuple[int, str]:
    """Returns (read locks left behind, the state B was in when aborted)."""
    prefix = tokens(PREFIX)
    a = new_id()
    add(a, prefix + tokens(8), 2)  # stores the prefix in LMCache
    drain()
    before = read_locked()

    fillers: list[str] = []
    if name == "abort while WAITING (lookup done, no free seat)":
        for _ in range(runner.cfg.max_num_seqs):  # occupy every seat, so B must wait
            f = new_id()
            add(f, tokens(40), 60)
            fillers.append(f)
        step()
        step()

    b = new_id()
    full_hit = name.startswith("no abort, full-prompt hit")
    add(b, prefix if full_hit else prefix + tokens(8), 8)
    seen = runner._request_state(b)

    if name == "abort right after submit (no step)":
        engine.abort_request([b])
    elif name == "abort while WAITING (lookup done, no free seat)":
        step()
        step()
        seen = runner._request_state(b)
        engine.abort_request([b])
    elif name.startswith("abort while"):
        target = "WAITING_FOR_REMOTE_KVS" if "loading" in name else "RUNNING"
        for _ in range(50):
            step()
            seen = runner._request_state(b)
            if seen == target:
                engine.abort_request([b])
                break
        else:
            seen = f"never reached {target} (last {seen})"
    for f in fillers:
        engine.abort_request([f])
    drain()
    return read_locked() - before, seen


CASES = [
    "no abort (control)",
    "no abort, full-prompt hit (control)",
    "abort right after submit (no step)",
    "abort while WAITING (lookup done, no free seat)",
    "abort while loading from LMCache (WAITING_FOR_REMOTE_KVS)",
    "abort while RUNNING",
]

print(f"LMCache read locks left behind once the engine is idle "
      f"({REPEATS} runs per case, fresh prefix each run)")
for name in CASES:
    results = [case(name) for _ in range(REPEATS)]
    left = [r[0] for r in results]
    states = sorted({r[1] for r in results})
    print(f"  {name:58s} left {left}  B's state at abort: {', '.join(states)}")
print(f"total read-locked now: {read_locked()}")
runner.lmcache.stop()
