"""Command line:  python -m phantomkv fuzz | replay"""

from __future__ import annotations

import argparse
import sys
import time

from .engine import PLANTS
from .sim import Config, run_seed


def _fuzz(a: argparse.Namespace) -> int:
    cfg = Config(remote_frac=a.remote_frac, abort_frac=a.abort_frac)
    t0, failing, kinds = time.perf_counter(), [], {}
    for seed in range(a.start, a.start + a.seeds):
        res = run_seed(seed, cfg, a.plant)
        if not res.ok:
            failing.append(res)
            for v in res.violations:
                kinds[v.kind] = kinds.get(v.kind, 0) + 1
            if a.stop_first:
                break
    dt = time.perf_counter() - t0
    ran = (failing[-1].seed - a.start + 1) if (a.stop_first and failing) else a.seeds
    print(f"phantomkv: {ran} seeds in {dt:.1f}s ({ran / dt:.0f} seeds/s), "
          f"plant={a.plant or 'none'}")
    if not failing:
        print("no violations found")
        return 0
    print(f"{len(failing)} failing seeds; violation kinds: {kinds}")
    first = failing[0]
    print(f"first failure: seed {first.seed}\n  {first.violations[0]}")
    print(f"replay with:  python -m phantomkv replay --seed {first.seed}"
          + (f" --plant {a.plant}" if a.plant else ""))
    return 1


def _replay(a: argparse.Namespace) -> int:
    res = run_seed(a.seed, Config(), a.plant)
    lines = res.trace
    if not a.full and not res.ok:
        # show the story around the first violation
        idx = next(i for i, l in enumerate(lines) if "VIOLATION" in l)
        lines = lines[max(0, idx - a.context): idx + 1]
    print("\n".join(lines))
    print(f"\nseed={a.seed} fingerprint={res.fingerprint} ticks={res.ticks} "
          f"finished={res.finished} aborted={res.aborted} violations={len(res.violations)}")
    return 0 if res.ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="phantomkv",
                                description="Deterministic simulation testing for LLM serving KV caches")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fuzz", help="run many seeds and report violations")
    f.add_argument("--seeds", type=int, default=500)
    f.add_argument("--start", type=int, default=0)
    f.add_argument("--plant", choices=PLANTS, default=None, help="switch on a known bug")
    f.add_argument("--remote-frac", type=float, default=0.5)
    f.add_argument("--abort-frac", type=float, default=0.25)
    f.add_argument("--stop-first", action="store_true")
    f.set_defaults(fn=_fuzz)

    r = sub.add_parser("replay", help="replay one seed and print its trace")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--plant", choices=PLANTS, default=None)
    r.add_argument("--context", type=int, default=25, help="trace lines before the violation")
    r.add_argument("--full", action="store_true")
    r.set_defaults(fn=_replay)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
