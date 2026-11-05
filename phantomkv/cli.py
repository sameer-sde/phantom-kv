"""Command line:  python -m phantomkv fuzz | replay | shrink | vllm-fuzz | vllm-shrink |
                                    pd-fuzz | pd-shrink"""

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
    ran = 0
    for seed in range(a.start, a.start + a.seeds):
        if runner.poisoned:
            print(f"vLLM raised an exception ({runner.poisoned}); its state can no longer "
                  f"be trusted, so seeds {seed}..{a.start + a.seeds - 1} were skipped")
            break
        res = runner.run(seed)
        ran += 1
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
    print(f"phantomkv vllm: {ran} seeds, {reqs} requests in {dt:.1f}s "
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
    phantom_found = any(r.violations or r.mismatches or r.liveness for r in failing)
    print(f"first failure: seed {first.seed}: {first.kinds}")
    for v in first.violations[:3]:
        print("  ", v)
    for m in first.mismatches[:3]:
        print("  ", m)
    if first.engine_error:
        print("   engine error:", first.engine_error)
    # exit 1: Phantom found violations; exit 2: only vLLM's own exceptions fired
    if not phantom_found:
        print("verdict: no Phantom violation; vLLM raised its own exception first")
        return 2
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


def _pd_config(a: argparse.Namespace):
    from .vllm_pd import PdConfig

    return PdConfig(pool_size=a.pool)


def _pd_prefill(a: argparse.Namespace) -> int:
    from .vllm_pd import run_prefill  # internal: the prefill engine's own process

    return run_prefill(_pd_config(a), a.storage)


def _pd_fuzz(a: argparse.Namespace) -> int:
    from .vllm_pd import PdRunner  # needs the vLLM environment

    t0 = time.perf_counter()
    runner = PdRunner(_pd_config(a), storage=a.storage, fault=a.fault, plant=a.plant)
    print(f"engines ready in {time.perf_counter() - t0:.1f}s "
          f"(pool {'prefilled now' if runner.prefilled else 'reused'})  "
          f"fault={a.fault or 'none'} plant={a.plant or 'none'}")
    t1, failing, kinds, ran = time.perf_counter(), [], {}, 0
    totals: dict[str, float] = {}
    for seed in range(a.start, a.start + a.seeds):
        if runner.poisoned:
            print(f"vLLM raised an exception ({runner.poisoned}); its state can no longer "
                  f"be trusted, so seeds {seed}..{a.start + a.seeds - 1} were skipped")
            break
        res = runner.run(seed)
        ran += 1
        for k, val in res.metrics.items():
            if k != "delay":
                totals[k] = totals.get(k, 0) + val
        if a.verbose:
            m = res.metrics
            print(f"seed {seed}: delay={m['delay']} loads={m['loads_started']} "
                  f"aborts={m['aborts']} (in flight {m['aborts_in_flight']}) "
                  f"-> {res.kinds or 'clean'}")
        if a.check_determinism and runner.run(seed).fingerprint != res.fingerprint:
            print(f"seed {seed}: NONDETERMINISTIC (two runs differ)")
            failing.append(res)
            continue
        if not res.ok:
            failing.append(res)
            for k, n in res.kinds.items():
                kinds[k] = kinds.get(k, 0) + n
    dt = time.perf_counter() - t1
    print(f"phantomkv pd: {ran} seeds in {dt:.1f}s")
    print(f"evidence: async loads started={int(totals.get('loads_started', 0))} "
          f"landed={int(totals.get('loads_landed', 0))} "
          f"aborts={int(totals.get('aborts', 0))} "
          f"aborts_while_kv_in_flight={int(totals.get('aborts_in_flight', 0))} "
          f"transferred_tokens_read={int(totals.get('transferred_tokens', 0))}")
    if not totals.get("aborts_in_flight"):
        print("warning: no abort happened while KV was in flight; the race was not exercised")
    if not failing:
        print("no violations found")
        return 0
    first = failing[0]
    print(f"{len(failing)} failing seeds; kinds: {kinds}")
    print(f"first failure: seed {first.seed} (delay {first.metrics.get('delay')}): {first.kinds}")
    for v in first.violations[:3]:
        print("  ", v)
    for m in first.mismatches[:3]:
        print("  ", m)
    if first.engine_error:
        print("   engine error:", first.engine_error)
    extra = "".join(f" --{k} {v}" for k, v in (("fault", a.fault), ("plant", a.plant)) if v)
    print(f"shrink with:  python -m phantomkv pd-shrink --seed {first.seed}{extra}")
    if not any(r.violations or r.mismatches or r.liveness for r in failing):
        print("verdict: no Phantom violation; vLLM raised its own exception first")
        return 2
    return 1


def _pd_shrink(a: argparse.Namespace) -> int:
    from .vllm_pd import PdRunner, pd_shrink  # needs the vLLM environment

    t0 = time.perf_counter()
    runner = PdRunner(_pd_config(a), storage=a.storage, fault=a.fault, plant=a.plant)
    print(f"engines ready in {time.perf_counter() - t0:.1f}s  "
          f"fault={a.fault or 'none'} plant={a.plant or 'none'}")
    t1 = time.perf_counter()
    res = pd_shrink(runner, a.seed)
    if res is None:
        print(f"seed {a.seed} passes; nothing to shrink")
        return 0
    print(f"shrunk seed {a.seed} (load delay {res.delay} steps): {len(res.original)} events "
          f"-> {len(res.minimal)} ({res.runs} vLLM runs, {time.perf_counter() - t1:.1f}s), "
          f"kind={res.kind}")
    print(f"reproduces {res.reproduced}/{res.confirm} times on fresh re-runs\n")
    print("minimal reproduction:")
    for ev in sorted(res.minimal, key=lambda e: (e.time, e.kind != "submit", e.rid)):
        line = f"  step {ev.time:<4} {ev.kind:<6} req {ev.rid:<3}"
        if ev.kind == "submit":
            src = "TRANSFERRED (KV from prefill engine)" if ev.remote else "local filler"
            line += f" prompt_len={len(ev.prompt):<4} max_new={ev.max_new:<3} {src}"
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

    def pd_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--fault", choices=["swap", "silent-drop"], default=None,
                        help="plant a transfer fault in the decode engine's connector")
        sp.add_argument("--plant", default=None,
                        help="plant a vLLM scheduler bug in the decode engine (e.g. abort-free-race)")
        sp.add_argument("--pool", type=int, default=16, help="prompts prefilled once and reused")
        sp.add_argument("--storage", default="pd_fuzz_storage",
                        help="shared KV storage between the two engines")

    pf = sub.add_parser("pd-fuzz", help="seeded fuzzing of two-engine prefill/decode on real vLLM")
    pf.add_argument("--seeds", type=int, default=20)
    pf.add_argument("--start", type=int, default=0)
    pf.add_argument("--check-determinism", action="store_true",
                    help="run every seed twice and require identical results")
    pf.add_argument("-v", "--verbose", action="store_true")
    pd_common(pf)
    pf.set_defaults(fn=_pd_fuzz)

    ps = sub.add_parser("pd-shrink", help="minimise a failing two-engine seed")
    ps.add_argument("--seed", type=int, required=True)
    pd_common(ps)
    ps.set_defaults(fn=_pd_shrink)

    pp = sub.add_parser("pd-prefill")  # internal: run by pd-fuzz in its own process
    pd_common(pp)
    pp.set_defaults(fn=_pd_prefill)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
