"""Drive real vLLM step by step from a seeded event list.

The same idea as sim.py, but the engine is vLLM's real LLMEngine:
  * seed -> list[Event] (submits with prompt contents, client aborts)
  * event.time is an engine-step index: at step k we add/abort exactly the
    events with time == k, then call engine.step() once
  * every request's output is compared with the pure-Python reference, and
    the Phantom model's provenance violations are collected per run

One engine is built per process and reused across seeds (building it takes
seconds); the prefix cache is reset between seeds so runs don't leak into
each other. A planted bug (PHANTOM_VLLM_PLANT) is fixed for the engine's life.

Only imported inside the vLLM environment; the core package never imports it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field

from .sim import Event
from .tags import expected_continuation

BLOCK_SIZE = 32  # default; profiles may change it


def page_bytes(block_size: int) -> int:
    return 2 * block_size * 1 * 32 * 4  # K+V x tokens x heads x dims x float32


@dataclass
class VllmConfig:
    num_requests: int = 24
    kv_blocks: int = 48
    block_size: int = BLOCK_SIZE
    max_num_seqs: int = 4
    max_model_len: int = 512
    max_num_batched_tokens: int | None = None  # small -> long prefills are chunked
    async_scheduling: bool = False
    shared_prefixes: int = 3
    max_body: int = 160
    max_new: int = 24
    abort_frac: float = 0.25
    arrival_window: int = 40  # engine steps
    # Second wave: requests that come back for the first whole blocks of an
    # earlier prompt after the cache has churned. Stale-cache bugs only bite
    # when an evicted prefix is requested again, which random traffic rarely does.
    revisit_requests: int = 10
    revisit_gap: int = 30  # wave 2 starts this many steps after wave 1 ends
    revisit_window: int = 40
    max_steps: int = 3_000


# Phase 3: each profile stresses a different part of vLLM's KV bookkeeping.
PROFILES: dict[str, dict] = {
    # Day 7/8 configuration: 48 x 32-token blocks, 4 concurrent sequences.
    "baseline": {},
    # Prefills split across steps: blocks are cached and hit while partially
    # computed requests are still mid-prompt.
    "chunked-prefill": {"max_num_batched_tokens": 64},
    # More concurrent sequences than the cache can hold: recompute preemption.
    "preemption": {"kv_blocks": 24, "max_num_seqs": 8},
    # Scheduler plans step N+1 while step N runs: frees/allocations interleave
    # with in-flight work.
    "async": {"async_scheduling": True},
    # Bigger blocks, same bytes: different block boundaries and eviction units.
    "block64": {"block_size": 64, "kv_blocks": 24},
}


def profile_config(name: str) -> "VllmConfig":
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    return VllmConfig(**PROFILES[name])


def vllm_workload(seed: int, cfg: VllmConfig) -> list[Event]:
    """Seed -> events sized for 32-token blocks (prefixes span whole blocks).

    Wave 1: `num_requests` arrivals with shared prefixes and client aborts.
    Wave 2: `revisit_requests` that reuse the first whole blocks of a random
    wave-1 prompt, after the cache has churned.
    """
    rng = random.Random(seed)
    prefixes = [[rng.randrange(1, 50_000) for _ in range(rng.choice((32, 64, 96)))]
                for _ in range(cfg.shared_prefixes)]
    events: list[Event] = []
    for rid in range(cfg.num_requests):
        body = [rng.randrange(1, 50_000) for _ in range(rng.randrange(1, cfg.max_body))]
        prompt = (rng.choice(prefixes) + body) if rng.random() < 0.6 else body
        max_new = 1 + rng.randrange(cfg.max_new)
        prompt = prompt[: cfg.max_model_len - max_new - 1]
        t = rng.randrange(cfg.arrival_window)
        events.append(Event(t, "submit", rid, tuple(prompt), max_new))
        if rng.random() < cfg.abort_frac:
            events.append(Event(t + rng.randrange(1, 30), "abort", rid))

    B = cfg.block_size
    wave1 = [e for e in events if e.kind == "submit" and len(e.prompt) > B]
    start = cfg.arrival_window + cfg.revisit_gap
    for i in range(cfg.revisit_requests if wave1 else 0):
        rid = cfg.num_requests + i
        src = rng.choice(wave1)
        k = rng.randrange(1, len(src.prompt) // B + 1)  # whole blocks
        body = [rng.randrange(1, 50_000) for _ in range(rng.randrange(1, 40))]
        max_new = 1 + rng.randrange(cfg.max_new)
        prompt = (list(src.prompt[: k * B]) + body)[: cfg.max_model_len - max_new - 1]
        events.append(Event(start + rng.randrange(cfg.revisit_window), "submit", rid,
                            tuple(prompt), max_new))
    return events


@dataclass
class VllmResult:
    seed: int
    steps: int
    finished: int
    aborted: int
    mismatches: list[str]
    violations: list[dict]
    liveness: bool
    outputs: dict[int, list[int]] = field(repr=False, default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)  # vLLM counters, this run

    @property
    def ok(self) -> bool:
        return not (self.mismatches or self.violations or self.liveness)

    @property
    def kinds(self) -> dict[str, int]:
        k: dict[str, int] = {}
        for v in self.violations:
            k[v["kind"]] = k.get(v["kind"], 0) + 1
        for _ in self.mismatches:
            k["output-mismatch"] = k.get("output-mismatch", 0) + 1
        if self.liveness:
            k["liveness"] = 1
        return k

    @property
    def fingerprint(self) -> str:
        blob = json.dumps({"out": sorted(self.outputs.items()),
                           "viol": [(v["kind"], v["detail"]) for v in self.violations],
                           "steps": self.steps}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class VllmRunner:
    def __init__(self, cfg: VllmConfig | None = None, plant: str | None = None,
                 report: str = "phantom_report.jsonl") -> None:
        self.cfg = cfg or VllmConfig()
        self.report = os.path.abspath(report)
        if os.path.exists(self.report):
            os.remove(self.report)
        os.environ["PHANTOM_REPORT"] = self.report
        if plant:
            os.environ["PHANTOM_VLLM_PLANT"] = plant  # inherited by EngineCore
        else:
            os.environ.pop("PHANTOM_VLLM_PLANT", None)
        self.plant = plant

        from vllm import LLM  # imported here: only available in the vLLM venv

        self.llm = LLM(
            model="facebook/opt-125m",
            hf_overrides={"architectures": ["PhantomForCausalLM"]},
            load_format="dummy",
            skip_tokenizer_init=True,
            dtype="float32",
            enforce_eager=True,
            async_scheduling=self.cfg.async_scheduling,
            enable_prefix_caching=True,
            block_size=self.cfg.block_size,
            max_model_len=self.cfg.max_model_len,
            max_num_seqs=self.cfg.max_num_seqs,
            kv_cache_memory_bytes=self.cfg.kv_blocks * page_bytes(self.cfg.block_size),
            seed=0,
            disable_log_stats=False,  # so get_metrics() can prove the chaos happened
            **self._extra_engine_args(),
        )
        self.engine = self.llm.llm_engine
        self.vocab = self.engine.model_config.hf_config.vocab_size

    def _extra_engine_args(self) -> dict:
        if self.cfg.max_num_batched_tokens is None:
            return {}
        return {"enable_chunked_prefill": True,
                "max_num_batched_tokens": self.cfg.max_num_batched_tokens}

    COUNTERS = ("vllm:num_preemptions", "vllm:prefix_cache_hits", "vllm:prefix_cache_queries")

    def counters(self) -> dict[str, float]:
        """Current values of the vLLM counters we report (summed over labels)."""
        out = {k: 0.0 for k in self.COUNTERS}
        try:
            for m in self.engine.get_metrics():
                for k in self.COUNTERS:
                    if m.name == k or m.name == k + "_total":
                        out[k] += float(getattr(m, "value", 0) or 0)
        except Exception:  # metrics are diagnostics; never fail a run on them
            pass
        return out

    def _report_offset(self) -> int:
        return os.path.getsize(self.report) if os.path.exists(self.report) else 0

    def _violations_since(self, offset: int) -> list[dict]:
        if not os.path.exists(self.report):
            return []
        with open(self.report) as f:
            f.seek(offset)
            return [json.loads(line) for line in f if line.strip()]

    def run(self, seed: int, events: list[Event] | None = None) -> VllmResult:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        events = vllm_workload(seed, self.cfg) if events is None else events
        self.engine.reset_prefix_cache()
        offset = self._report_offset()
        before = self.counters()
        by_step: dict[int, list[Event]] = {}
        for ev in events:
            by_step.setdefault(ev.time, []).append(ev)
        last_event_step = max(by_step, default=-1)

        prompts: dict[int, tuple[int, ...]] = {}
        outputs: dict[int, list[int]] = {}
        live: set[int] = set()
        finished: set[int] = set()
        aborted: set[int] = set()
        step = 0
        while step <= last_event_step or live:
            for ev in by_step.get(step, ()):
                if ev.kind == "submit":
                    params = SamplingParams(max_tokens=ev.max_new, temperature=0,
                                            detokenize=False)
                    self.engine.add_request(str(ev.rid),
                                            TokensPrompt(prompt_token_ids=list(ev.prompt)),
                                            params)
                    prompts[ev.rid] = ev.prompt
                    live.add(ev.rid)
                elif ev.kind == "abort" and ev.rid in live:
                    self.engine.abort_request([str(ev.rid)])
                    live.discard(ev.rid)
                    aborted.add(ev.rid)
            if live:
                for out in self.engine.step():
                    rid = int(str(out.request_id).split("-")[0])
                    if rid in aborted:
                        continue
                    outputs[rid] = list(out.outputs[0].token_ids)
                    if out.finished:
                        live.discard(rid)
                        finished.add(rid)
            step += 1
            if step > self.cfg.max_steps:
                break

        mismatches = []
        for rid, got in sorted(outputs.items()):
            want = expected_continuation(list(prompts[rid]), len(got), self.vocab)
            if got != want:
                mismatches.append(f"req {rid}: first diff at output token "
                                  f"{next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)}")
        after = self.counters()
        metrics = {k.split(":")[1]: after[k] - before[k] for k in self.COUNTERS}
        return VllmResult(seed, step, len(finished), len(aborted), mismatches,
                          self._violations_since(offset), liveness=bool(live), outputs=outputs,
                          metrics=metrics)


# ------------------------------------------------------------------ shrinking
def primary_kind(res: VllmResult) -> str | None:
    """The failure kind a shrink must preserve (first violation wins)."""
    if res.violations:
        return res.violations[0]["kind"]
    if res.mismatches:
        return "output-mismatch"
    if res.liveness:
        return "liveness"
    return None


@dataclass
class VllmShrinkResult:
    seed: int
    kind: str
    original: list[Event]
    minimal: list[Event]
    result: VllmResult
    runs: int
    reproduced: int  # how many of `confirm` fresh re-runs of `minimal` still failed
    confirm: int


def shrink(runner: VllmRunner, seed: int, confirm: int = 3) -> VllmShrinkResult | None:
    """ddmin over the event list, re-running real vLLM for every candidate.

    The engine is reused (start-up takes ~80 s; one run takes well under a
    second); the prefix cache is reset before every run. The minimal list is
    then re-run `confirm` times to check that it reproduces reliably.
    """
    from .shrink import ddmin

    events = vllm_workload(seed, runner.cfg)
    first = runner.run(seed, events)
    kind = primary_kind(first)
    if kind is None:
        return None
    runs = 1

    def fails(evs: list[Event]) -> bool:
        nonlocal runs
        runs += 1
        res = runner.run(seed, evs)
        return kind in res.kinds

    minimal = ddmin(events, fails)
    finals = [runner.run(seed, minimal) for _ in range(confirm)]
    reproduced = sum(kind in r.kinds for r in finals)
    return VllmShrinkResult(seed, kind, events, minimal, finals[0], runs, reproduced, confirm)
