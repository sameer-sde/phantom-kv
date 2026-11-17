"""Replay saved minimal LMCache lock-leak cases (Day 24).

Each JSON file is an event list saved by scripts/lmcache_lock_shrink.py. Every
case runs 3 times; a forced clear between runs empties LMCache, locked objects
included. Prints the read locks each run left behind (0 = no leak).

Run in the vLLM + LMCache env:
    python scripts/lmcache_lock_replay.py tests/fixtures/lmcache_lock_leak_seed6.json \\
        tests/fixtures/lmcache_lock_leak_seed9.json
"""

from __future__ import annotations

import json
import os
import sys

os.environ["PHANTOM_LMCACHE_FORCE_CLEAR"] = "1"  # read by LMCacheServer.clear()

from phantomkv.sim import Event  # noqa: E402
from phantomkv.vllm_harness import VllmRunner, profile_config  # noqa: E402

runner = VllmRunner(profile_config("lmcache-only"))
for path in sys.argv[1:]:
    with open(path) as f:
        events = [Event(**{**e, "prompt": tuple(e["prompt"])}) for e in json.load(f)]
    seed = int("".join(c for c in os.path.basename(path) if c.isdigit()) or 0)
    left = []
    for _ in range(3):
        m = runner.run(seed, events).metrics
        left.append(m["lmcache_read_locked_before_clear"] - m["lmcache_read_locked_at_start"])
    print(f"LOCKS {os.path.basename(path)}: read locks left on 3 runs {left}", flush=True)
runner.lmcache.stop()
os._exit(0)  # vLLM's engine shutdown can hang at interpreter exit; results are printed
