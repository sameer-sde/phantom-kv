"""Command line:  python -m phantomkv fuzz | replay | shrink | vllm-fuzz"""

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


def _shrink(a: argparse.Namespace) -> int:
    from .shrink import shrink_seed

    t0 = time.perf_counter()
    res = shrink_seed(a.seed, Config(), a.plant)
    if res is None:
        print(f"seed {a.seed} passes; nothing to shrink")
        return 0
    dt = time.perf_counter() - t0
    print(f"shrunk seed {a.seed}: {len(res.original)} events -> {len(res.minimal)} "
          f"({res.runs} runs, {dt:.2f}s)\n")
    print("minimal reproduction:")
    for ev in sorted(res.minimal, key=lambda e: (e.time, e.rid)):
        print(f"  {ev}")
    print("\ntrace:")
    print("\n".join(f"  {line}" for line in res.result.trace))
    print(f"\n=> {res.violation}")
    return 1


def _vllm_fuzz(a: argparse.Namespace) -> int:
    from .vllm_harness import VllmRunner  # needs the vLLM environment

    t0 = time.perf_counter()
    runner = VllmRunner(plant=a.plant)
    print(f"engine ready in {time.perf_counter() - t0:.1f}s  plant={a.plant or 'none'}")
    t1, failing, kinds, reqs = time.perf_counter(), [], {}, 0
    for seed in range(a.start, a.start + a.seeds):
        res = runner.run(seed)
        reqs += res.finished + res.aborted
        if a.check_determinism and runner.run(seed).fingerprint != res.fingerprint:
            print(f"seed {seed}: NONDETERMINISTIC (two runs differ)")
            failing.append(res)
            continue
        if not res.ok:
            failing.append(res)
            for k, n in res.kinds.items():
                kinds[k] = kinds.get(k, 0) + n
            if a.verbose:
                print(f"seed {seed}: {res.kinds}")
    dt = time.perf_counter() - t1
    print(f"phantomkv vllm: {a.seeds} seeds, {reqs} requests in {dt:.1f}s "
          f"({reqs / dt:.1f} req/s)")
    if not failing:
        print("no violations found")
        return 0
    first = failing[0]
    print(f"{len(failing)} failing seeds; kinds: {kinds}")
    print(f"first failure: seed {first.seed}: {first.kinds}")
    for v in first.violations[:3]:
        print("  ", v)
    for m in first.mismatches[:3]:
        print("  ", m)
    return 1


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

    s = sub.add_parser("shrink", help="minimise a failing seed to a small reproduction")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--plant", choices=PLANTS, default=None)
    s.set_defaults(fn=_shrink)

    v = sub.add_parser("vllm-fuzz", help="fuzz real vLLM step by step (needs the vLLM env)")
    v.add_argument("--seeds", type=int, default=20)
    v.add_argument("--start", type=int, default=0)
    v.add_argument("--plant", default=None, help="plant a known bug into vLLM")
    v.add_argument("--check-determinism", action="store_true",
                   help="run every seed twice and require identical results")
    v.add_argument("-v", "--verbose", action="store_true")
    v.set_defaults(fn=_vllm_fuzz)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
