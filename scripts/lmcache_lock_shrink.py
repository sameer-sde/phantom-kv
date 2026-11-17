"""Shrink a seed that leaves LMCache read locks behind (Day 24).

A single aborted request never leaks a lock (scripts/lmcache_abort_probe.py),
but some fuzz seeds do. This runs Phantom's ddmin shrinker over a seed's event
list with a different failure test: "after every request is done, LMCache
holds read locks this run added". Between candidates a *forced* clear empties
LMCache, locked objects included, so each candidate starts from nothing.

Run in the vLLM + LMCache env:  python scripts/lmcache_lock_shrink.py 6 9
"""

from __future__ import annotations

import os
import sys

os.environ["PHANTOM_LMCACHE_FORCE_CLEAR"] = "1"  # read by LMCacheServer.clear()

from phantomkv.shrink import ddmin  # noqa: E402
from phantomkv.vllm_harness import VllmRunner, profile_config, vllm_workload  # noqa: E402

CHUNK = 32
runner = VllmRunner(profile_config("lmcache-only"))


def new_locks(seed: int, events) -> int:
    res = runner.run(seed, list(events))
    m = res.metrics
    return m["lmcache_read_locked_before_clear"] - m["lmcache_read_locked_at_start"]


def shared_chunks(a: tuple, b: tuple) -> int:
    n = 0
    while (n + 1) * CHUNK <= min(len(a), len(b)) and \
            a[n * CHUNK:(n + 1) * CHUNK] == b[n * CHUNK:(n + 1) * CHUNK]:
        n += 1
    return n


def describe(events) -> None:
    prompts = {e.rid: e.prompt for e in events if e.kind == "submit"}
    for e in sorted(events, key=lambda e: (e.time, e.kind != "submit")):
        if e.kind == "abort":
            print(f"    step {e.time:3d}  abort  req {e.rid}")
            continue
        shares = [f"{shared_chunks(e.prompt, p)} with req {r}" for r, p in prompts.items()
                  if r != e.rid and shared_chunks(e.prompt, p)]
        print(f"    step {e.time:3d}  submit req {e.rid}: {len(e.prompt)} tokens "
              f"({len(e.prompt) // CHUNK} whole chunks), max_new {e.max_new}"
              + (f"; shares chunks: {', '.join(shares)}" if shares else ""))


for seed in [int(s) for s in sys.argv[1:]] or [6]:
    events = vllm_workload(seed, runner.cfg)
    first = new_locks(seed, events)
    print(f"\n=== seed {seed}: {len(events)} events, {first} read locks left behind")
    if not first:
        print("    no leak on this run; nothing to shrink")
        continue
    runs = 0

    def leaks(evs) -> bool:
        global runs
        runs += 1
        return new_locks(seed, evs) > 0

    minimal = ddmin(events, leaks)
    again = [new_locks(seed, minimal) for _ in range(5)]
    print(f"    shrunk to {len(minimal)} events in {runs} runs; "
          f"locks left on 5 re-runs: {again}")
    describe(minimal)
runner.lmcache.stop()
