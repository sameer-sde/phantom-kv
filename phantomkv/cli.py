"""Command line:  python -m phantomkv fuzz | replay | shrink | vllm-fuzz | vllm-shrink"""

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
    from .vllm_harness import VllmRunner, profile_config  # needs the vLLM environment

    t0 = time.perf_counter()
    runner = VllmRunner(profile_config(a.profile), plant=a.plant)
    print(f"engine ready in {time.perf_counter() - t0:.1f}s  "
          f"profile={a.profile} plant={a.plant or 'none'}")
    t1, failing, kinds, reqs = time.perf_counter(), [], {}, 0
    totals: dict[str, float] = {}
    for seed in range(a.start, a.start + a.seeds):
        res = runner.run(seed)
        reqs += res.finished + res.aborted
        for k, val in res.metrics.items():
            if k == "max_step_tokens":
                totals[k] = max(totals.get(k, 0), val)
            else:
                totals[k] = totals.get(k, 0) + val
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
    if totals:
        hits, queries = totals.get("prefix_cache_hits", 0), totals.get("prefix_cache_queries", 0)
        rate = f" ({100 * hits / queries:.0f}% of queried tokens)" if queries else ""
        print(f"engine counters: preemptions={int(totals.get('num_preemptions', 0))} "
              f"prefix_cache_hit_tokens={int(hits)}{rate} "
              f"chunked_prefill_continuations={int(totals.get('prefill_continuations', 0))} "
              f"max_tokens_per_step={int(totals.get('max_step_tokens', 0))}")
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
    if first.engine_error:
        print("   engine error:", first.engine_error)
    return 1


def _vllm_shrink(a: argparse.Namespace) -> int:
    from .vllm_harness import VllmRunner, profile_config, shrink  # needs the vLLM env

    t0 = time.perf_counter()
    runner = VllmRunner(profile_config(a.profile), plant=a.plant)
    print(f"engine ready in {time.perf_counter() - t0:.1f}s  plant={a.plant or 'none'}")
    t1 = time.perf_counter()
    res = shrink(runner, a.seed)
    if res is None:
        print(f"seed {a.seed} passes; nothing to shrink")
        return 0
    print(f"shrunk seed {a.seed}: {len(res.original)} events -> {len(res.minimal)} "
          f"({res.runs} vLLM runs, {time.perf_counter() - t1:.1f}s), kind={res.kind}")
    print(f"reproduces {res.reproduced}/{res.confirm} times on fresh re-runs\n")
    groups: dict[tuple[int, ...], str] = {}
    print("minimal reproduction (same letter = same first 32-token block):")
    for ev in sorted(res.minimal, key=lambda e: (e.time, e.rid)):
        line = f"  step {ev.time:<4} {ev.kind:<6} req {ev.rid:<3}"
        if ev.kind == "submit":
            g = groups.setdefault(ev.prompt[:32], "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[len(groups) % 26])
            line += f" prompt_len={len(ev.prompt):<4} first_block={g} max_new={ev.max_new}"
        print(line)
    print("\nviolations in the minimal run:")
    for v in res.result.violations[:5]:
        print("  ", v)
    for m in res.result.mismatches[:5]:
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
    v.add_argument("--profile", default="baseline",
                   help="vLLM configuration: baseline, chunked-prefill, preemption, block64")
    v.add_argument("--check-determinism", action="store_true",
                   help="run every seed twice and require identical results")
    v.add_argument("-v", "--verbose", action="store_true")
    v.set_defaults(fn=_vllm_fuzz)

    vs = sub.add_parser("vllm-shrink", help="minimise a failing real-vLLM seed")
    vs.add_argument("--seed", type=int, required=True)
    vs.add_argument("--plant", default=None, help="plant a known bug into vLLM")
    vs.add_argument("--profile", default="baseline")
    vs.set_defaults(fn=_vllm_shrink)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
