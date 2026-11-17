"""Shrink a seed that leaves LMCache read locks behind (Day 24).

A single aborted request never leaks a lock (scripts/lmcache_abort_probe.py),
but some fuzz seeds do. This runs Phantom's ddmin shrinker over a seed's event
list with a different failure test: "after every request is done, LMCache
holds read locks this run added". Between candidates a *forced* clear empties
LMCache, locked objects included, so each candidate starts from nothing.

Run in the vLLM + LMCache env:  python scripts/lmcache_lock_shrink.py 6 9
"""

from __future__ import annotations

import dataclasses
import json
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


def explain(seed: int, events) -> None:
    """Re-run the minimal events once, logging LMCache's view of each aborted
    request at the moment of the abort."""
    engine = runner.engine
    real_abort = engine.abort_request

    def logged_abort(ids, *args, **kwargs):
        for rid in ids:
            sched = runner._scheduler()
            conn = getattr(sched, "connector", None)
            key = next((k for k in (getattr(sched, "requests", None) or {})
                        if k == rid or k.startswith(rid + "-")), None)
            tracker = (getattr(conn, "request_trackers", None) or {}).get(key)
            lookups = getattr(getattr(conn, "scheduler_adapter", None), "_lookups", None) or {}
            lk = lookups.get(key)
            print(f"    at abort of req {rid}: vLLM state {runner._request_state(rid)}, "
                  f"connector state {getattr(getattr(tracker, 'state', None), 'name', None)}, "
                  f"LMCache hit tokens {getattr(tracker, 'num_lmcache_hit_tokens', None)}, "
                  f"lookup {'none' if lk is None else 'pending' if getattr(lk, '_result', None) is None else 'returned ' + str(lk._result)}, "
                  f"read-locked before {runner.lmcache.l1_status()['read_locked']}")
        out = real_abort(ids, *args, **kwargs)
        runner._settle_lmcache()
        print(f"    read-locked right after the abort {runner.lmcache.l1_status()['read_locked']}")
        return out

    engine.abort_request = logged_abort
    try:
        print(f"    replay with logging: {new_locks(seed, events)} read locks left behind")
    finally:
        engine.abort_request = real_abort


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
    out = f"lmcache_lock_leak_seed{seed}.json"
    with open(out, "w") as f:
        json.dump([dataclasses.asdict(e) for e in minimal], f)
    print(f"    minimal events saved to {out}")
    explain(seed, minimal)
runner.lmcache.stop()
