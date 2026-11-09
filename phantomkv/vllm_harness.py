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
    # Phase 5: vLLM's SimpleCPUOffloadConnector ("eager" | "lazy"), with a
    # CPU cache of offload_cpu_blocks blocks; transfers complete offload_delay
    # engine steps after they are issued
    offload: str | None = None
    offload_cpu_blocks: int = 96
    offload_delay: int = 0
    offload_store_delay: int | None = None  # per-direction overrides
    offload_load_delay: int | None = None
    # Phase 5 target 2 ("tiered": vLLM's OffloadingConnector): GPU blocks per
    # offloaded chunk
    offload_chunk_blocks: int = 1
    offload_sync_budget: int = 0  # bytes per step of synchronous (expedited) loads
    # Phase 5 target 3 (offload="lmcache"): LMCache server L1 size, and whether
    # vLLM's own GPU prefix cache is on (off: reuse can only come from LMCache)
    lmcache_l1_gb: float = 1.0
    gpu_prefix_cache: bool = True
    lmcache_restart_every: int = 0  # restart the LMCache server every N runs (0: never)


# Phase 3: each profile stresses a different part of vLLM's KV bookkeeping.
PROFILES: dict[str, dict] = {
    # Day 7/8 configuration: 48 x 32-token blocks, 4 concurrent sequences.
    "baseline": {},
    # Prefills split across steps: blocks are cached and hit while partially
    # computed requests are still mid-prompt.
    "chunked-prefill": {"max_num_batched_tokens": 64},
    # More concurrent sequences than the cache can hold: recompute preemption.
    "preemption": {"kv_blocks": 24, "max_num_seqs": 8},
    # NOTE: no "async" profile. vLLM's CPU platform forces
    # scheduler_config.async_scheduling = False (vllm/platforms/cpu.py), so it
    # would silently be baseline again; async scheduling needs a GPU run.
    # Bigger blocks, same bytes: different block boundaries and eviction units.
    "block64": {"block_size": 64, "kv_blocks": 24},
    # Phase 5: CPU offloading. GPU blocks evicted from the prefix cache are
    # stored in a CPU cache and loaded back when a later request needs them.
    "cpu-offload": {"offload": "eager"},
    "cpu-offload-lazy": {"offload": "lazy"},
    # transfers take 2 steps: blocks the manager releases early get overwritten
    "cpu-offload-slow": {"offload": "eager", "offload_delay": 2},
    # small GPU and CPU caches: CPU-side LRU eviction
    "cpu-offload-small": {"offload": "eager", "kv_blocks": 32, "offload_cpu_blocks": 32},
    # transfers take 25 steps: still in flight when their requests finish and
    # the revisit wave arrives
    "cpu-offload-very-slow": {"offload": "eager", "offload_delay": 25},
    # more sequences than the GPU cache holds: preemption flushes (need_flush)
    "cpu-offload-preempt": {"offload": "eager", "kv_blocks": 24, "max_num_seqs": 8},
    # stores take 40 steps, loads 1: on separate queues (like vLLM's two CUDA
    # streams) a load can overtake a store; the manager must never let it
    # read a CPU block whose store has not completed
    "cpu-offload-slow-stores": {"offload": "eager", "offload_store_delay": 40,
                                "offload_load_delay": 1},
    # Phase 5 target 2: vLLM's OffloadingConnector (scheduler + CPUOffloadingManager
    # unmodified; only the CUDA copy engine is replaced, see vllm_offload2.py)
    "offload2": {"offload": "tiered"},
    # chunks of 2 GPU blocks: partial first chunks, alignment
    "offload2-chunk2": {"offload": "tiered", "offload_chunk_blocks": 2},
    "offload2-slow-stores": {"offload": "tiered", "offload_store_delay": 40,
                             "offload_load_delay": 1},
    "offload2-preempt": {"offload": "tiered", "kv_blocks": 24, "max_num_seqs": 8},
    "offload2-small": {"offload": "tiered", "kv_blocks": 32, "offload_cpu_blocks": 32},
    # synchronous loads: a request whose KV fits the per-step budget loads and
    # runs in the same step (expedited load + wait inside the step)
    "offload2-sync": {"offload": "tiered", "offload_sync_budget": 1 << 20},
    # partial chunks: fewer GPU blocks (more evictions) and 3x the revisits,
    # which come back for a random number of whole blocks, so an odd number
    # often ends part-way into a chunk
    "offload2-chunk2-revisit": {"offload": "tiered", "offload_chunk_blocks": 2,
                                "kv_blocks": 32, "revisit_requests": 30,
                                "revisit_window": 60},
    "offload2-chunk3-revisit": {"offload": "tiered", "offload_chunk_blocks": 3,
                                "kv_blocks": 32, "revisit_requests": 30,
                                "revisit_window": 60},
    # Phase 5 target 3: LMCache (separate CPU-only server; LMCache's own connector)
    "lmcache": {"offload": "lmcache"},
    # vLLM's own prefix cache off: every reused prefix must come from LMCache
    "lmcache-only": {"offload": "lmcache", "gpu_prefix_cache": False},
    "lmcache-preempt": {"offload": "lmcache", "kv_blocks": 24, "max_num_seqs": 8},
    # ~1 MB of LMCache memory: LMCache-side eviction under load
    "lmcache-small": {"offload": "lmcache", "gpu_prefix_cache": False, "lmcache_l1_gb": 0.001},
    # the LMCache server is killed and restarted between runs, losing everything
    "lmcache-restart": {"offload": "lmcache", "gpu_prefix_cache": False,
                        "lmcache_restart_every": 1},
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
    engine_error: str | None = None  # exception raised by vLLM during the run
    # per-step trace when the runner traces (`phantomkv vllm-diff`): (key, detail)
    # pairs; keys omit block ids, which may legitimately differ between runs
    trace: list[tuple[str, str]] = field(repr=False, default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.mismatches or self.violations or self.liveness or self.engine_error)

    @property
    def kinds(self) -> dict[str, int]:
        k: dict[str, int] = {}
        for v in self.violations:
            k[v["kind"]] = k.get(v["kind"], 0) + 1
        for _ in self.mismatches:
            k["output-mismatch"] = k.get("output-mismatch", 0) + 1
        if self.liveness:
            k["liveness"] = 1
        if self.engine_error:
            k["engine-error"] = 1
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
        for path in (self.report, self.report + ".stats.json"):  # stale files would skew deltas
            if os.path.exists(path):
                os.remove(path)
        self.poisoned: str | None = None  # set once vLLM raises: its state can't be trusted
        self.tracing = False  # set by `phantomkv vllm-diff`
        self.resets_refused = 0  # runs that started without a clean cache
        os.environ["PHANTOM_REPORT"] = self.report
        # Run vLLM's EngineCore in THIS process. With the default multiprocess
        # engine, the core runs its own busy loop: add/abort messages arrive
        # asynchronously, so "abort at step k" can land a step late (observed:
        # a fresh engine produced one extra token for an aborted request on its
        # first run only). In-process, add_request / abort_request / step()
        # execute in strict order, which deterministic replay requires.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        if plant:
            os.environ["PHANTOM_VLLM_PLANT"] = plant  # read by the plugin at engine start
        else:
            os.environ.pop("PHANTOM_VLLM_PLANT", None)
        self.plant = plant
        if self.cfg.offload in ("eager", "lazy"):
            os.environ["PHANTOM_OFFLOAD"] = "1"  # plugin installs PhantomOffloadWorker
            os.environ["PHANTOM_OFFLOAD_DELAY"] = str(self.cfg.offload_delay)
            for kind in ("store", "load"):
                val = getattr(self.cfg, f"offload_{kind}_delay")
                if val is None:
                    os.environ.pop(f"PHANTOM_OFFLOAD_{kind.upper()}_DELAY", None)
                else:
                    os.environ[f"PHANTOM_OFFLOAD_{kind.upper()}_DELAY"] = str(val)
        else:
            os.environ.pop("PHANTOM_OFFLOAD", None)
        self.lmcache = None
        if self.cfg.offload == "lmcache":
            from .lmcache_server import LMCacheServer

            self.lmcache = LMCacheServer(l1_gb=self.cfg.lmcache_l1_gb,
                                         chunk_size=self.cfg.block_size)
            self.lmcache.start()
        if self.cfg.offload == "tiered":  # read by vllm_offload2.PhantomOffloadWorker
            for kind in ("store", "load"):
                val = getattr(self.cfg, f"offload_{kind}_delay")
                os.environ[f"PHANTOM_OFFLOAD2_{kind.upper()}_DELAY"] = str(
                    self.cfg.offload_delay if val is None else val)

        from vllm import LLM  # imported here: only available in the vLLM venv

        self.llm = LLM(
            model="facebook/opt-125m",
            hf_overrides={"architectures": ["PhantomForCausalLM"]},
            load_format="dummy",
            skip_tokenizer_init=True,
            dtype="float32",
            enforce_eager=True,
            async_scheduling=self.cfg.async_scheduling,
            enable_prefix_caching=self.cfg.gpu_prefix_cache,
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
        args: dict = {}
        if self.cfg.max_num_batched_tokens is not None:
            args.update(enable_chunked_prefill=True,
                        max_num_batched_tokens=self.cfg.max_num_batched_tokens)
        if self.cfg.offload == "lmcache":
            from vllm.config import KVTransferConfig

            args["kv_transfer_config"] = KVTransferConfig(
                kv_connector="LMCacheMPConnector",  # LMCache's own connector, unmodified
                kv_connector_module_path="lmcache.integration.vllm.lmcache_mp_connector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "lmcache.mp.host": "tcp://localhost",
                    "lmcache.mp.port": self.lmcache.port,
                    "lmcache.mp.mp_transfer_mode": "engine_driven",
                })
        elif self.cfg.offload == "tiered":
            import mmap

            from vllm.config import KVTransferConfig

            # vLLM sizes the CPU cache from bytes: each chunk is rounded up to
            # the OS page size, so ask for exactly offload_cpu_blocks chunks
            chunk = self.cfg.offload_chunk_blocks * page_bytes(self.cfg.block_size)
            aligned = -(-chunk // mmap.PAGESIZE) * mmap.PAGESIZE
            args["kv_transfer_config"] = KVTransferConfig(
                kv_connector="OffloadingConnector",  # vLLM's own, unmodified
                kv_role="kv_both",
                kv_connector_extra_config={
                    "spec_name": "PhantomCPUOffloadingSpec",
                    "spec_module_path": "phantomkv.vllm_offload2",
                    "cpu_bytes_to_use": self.cfg.offload_cpu_blocks * aligned,
                    "blocks_per_chunk": self.cfg.offload_chunk_blocks,
                    "sync_load_budget_bytes": self.cfg.offload_sync_budget,
                })
        elif self.cfg.offload:
            from vllm.config import KVTransferConfig

            args["kv_transfer_config"] = KVTransferConfig(
                kv_connector="SimpleCPUOffloadConnector",  # vLLM's own, unmodified
                kv_role="kv_both",
                kv_connector_extra_config={
                    "num_cpu_blocks": self.cfg.offload_cpu_blocks,
                    "cpu_bytes_to_use": 1 << 30,  # only an upper bound here
                    "lazy_offload": self.cfg.offload == "lazy",
                })
        return args

    def _xfer(self):
        """The copy-engine module for this profile (STATS, TRACE, pending_transfers)."""
        if self.cfg.offload == "tiered":
            from . import vllm_offload2 as mod
        else:
            from . import vllm_offload as mod
        return mod

    def offload_stats(self) -> dict[str, int]:
        if self.cfg.offload == "lmcache":
            return dict(self.lmcache.counts(), server_restarts=self.lmcache.restarts)
        if not self.cfg.offload:
            return {}
        return dict(self._xfer().STATS)

    @property
    def has_copy_engine(self) -> bool:
        """Our own copy engine is in the loop (vLLM offloading targets)."""
        return self.cfg.offload in ("eager", "lazy", "tiered")

    COUNTERS = ("vllm:num_preemptions", "vllm:prefix_cache_hits", "vllm:prefix_cache_queries",
                "vllm:external_prefix_cache_hits", "vllm:external_prefix_cache_queries")

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

    def model_stats(self) -> dict[str, float]:
        """Cumulative counters written by the Phantom model (its stats file)."""
        try:
            with open(self.report + ".stats.json") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _report_offset(self) -> int:
        return os.path.getsize(self.report) if os.path.exists(self.report) else 0

    def _violations_since(self, offset: int) -> list[dict]:
        if not os.path.exists(self.report):
            return []
        with open(self.report) as f:
            f.seek(offset)
            return [json.loads(line) for line in f if line.strip()]

    KEEPALIVE_RID = 99_999

    def _settle_lmcache(self, timeout: float = 2.0) -> None:
        """Give LMCache's asynchronous RPCs time to complete before each step.

        The harness steps far faster than a real engine (no model compute), so
        LMCache's lookup replies, which arrive on background threads, never
        landed within the step budget: every request stayed WAITING (Day 22).
        Before each step, wait (bounded) until every pending lookup has a
        result, then a short pause for other in-flight RPCs (stores,
        retrievals). Lookups then resolve within one step, a deterministic
        schedule a fast local server can produce."""
        import time

        conn = getattr(self._scheduler(), "connector", None)
        lookups = getattr(getattr(conn, "scheduler_adapter", None), "_lookups", None) or {}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and any(
                getattr(lk, "_result", 0) is None for lk in list(lookups.values())):
            time.sleep(0.0005)
        time.sleep(float(os.environ.get("PHANTOM_LMCACHE_STEP_MS", "2")) / 1000)

    def _engine_busy(self) -> bool:
        sched = self._scheduler()
        return bool(sched is not None and sched.has_requests())

    def _scheduler(self):
        """vLLM's scheduler (in-process engine only), for tracing."""
        core = getattr(getattr(self.engine, "engine_core", None), "engine_core", None)
        return getattr(core, "scheduler", None)

    def max_offload_delay(self) -> int:
        return max(d for d in (self.cfg.offload_delay, self.cfg.offload_store_delay,
                               self.cfg.offload_load_delay) if d is not None)

    def _drain_offload(self, seed, step, prompts, outputs, finished, trace=None):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        pending_transfers = self._xfer().pending_transfers

        for k in range(3):
            if not pending_transfers():
                break
            rid = self.KEEPALIVE_RID - k
            prompt = tuple(random.Random(seed * 7 + k).randrange(1, 50_000) for _ in range(20))
            self.engine.add_request(str(rid), TokensPrompt(prompt_token_ids=list(prompt)),
                                    SamplingParams(max_tokens=self.max_offload_delay() + 2,
                                                   temperature=0, detokenize=False))
            prompts[rid] = prompt
            if trace is not None:
                trace.append((f"{step} drain: keep-alive {rid}, "
                              f"{pending_transfers()} transfers pending", ""))
            for _ in range(self.max_offload_delay() + 20):
                try:
                    outs = self.engine.step()
                except Exception as e:  # vLLM raised while draining
                    self.poisoned = f"seed {seed}, drain: {type(e).__name__}: {e}"[:400]
                    return step, self.poisoned
                if trace is not None:
                    xfer = self._xfer()
                    for tick, phase, kind, event, n, blocks in xfer.TRACE:
                        trace.append((f"{step} {kind} {phase} n={n}",
                                      f"event={event} src->dst={blocks}"))
                    xfer.TRACE.clear()
                step += 1
                done = False
                for out in outs:
                    if int(str(out.request_id).split("-")[0]) == rid:
                        outputs[rid] = list(out.outputs[0].token_ids)
                        done = out.finished
                if done:
                    finished.add(rid)
                    break
        return step, None

    def run(self, seed: int, events: list[Event] | None = None) -> VllmResult:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        events = vllm_workload(seed, self.cfg) if events is None else events
        # with CPU offloading, the connector's CPU cache must be reset too, or a
        # replay of the same seed finds blocks the first run stored
        if self.lmcache is not None:
            restart = self.cfg.lmcache_restart_every
            self.runs_started = getattr(self, "runs_started", 0) + 1
            if restart and self.runs_started > 1 and (self.runs_started - 1) % restart == 0:
                self.lmcache.restart()  # fault: the store loses everything
            elif not self.lmcache.clear():  # replays must start from an empty store
                self.resets_refused += 1
        reset_connector = bool(self.cfg.offload)
        if self.lmcache is not None and os.environ.get("PHANTOM_LMCACHE_CONNECTOR_RESET") == "0":
            reset_connector = False  # diagnostic: clear LMCache over HTTP only
        reset_ok = self.engine.reset_prefix_cache(reset_connector=reset_connector)
        offset = self._report_offset()
        before = self.counters()
        mbefore = self.model_stats()
        obefore = self.offload_stats()
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
        engine_error: str | None = None
        trace: list[tuple[str, str]] = []
        xfer = None
        if self.tracing:
            trace.append(("start", f"reset_prefix_cache -> {reset_ok}"))
            if self.has_copy_engine:
                xfer = self._xfer()
                xfer.TRACE_ON = True
                xfer.TRACE.clear()
        while step <= last_event_step or live or self._engine_busy():
            try:
                for ev in by_step.get(step, ()):
                    if self.tracing and (ev.kind == "submit" or ev.rid in live):
                        trace.append((f"{step} {ev.kind} {ev.rid}",
                                      f"prompt_len={len(ev.prompt)}" if ev.prompt else ""))
                    if ev.kind == "submit":
                        params = SamplingParams(max_tokens=ev.max_new, temperature=0,
                                                detokenize=False)
                        self.engine.add_request(
                            str(ev.rid), TokensPrompt(prompt_token_ids=list(ev.prompt)), params)
                        prompts[ev.rid] = ev.prompt
                        live.add(ev.rid)
                    elif ev.kind == "abort" and ev.rid in live:
                        self.engine.abort_request([str(ev.rid)])
                        live.discard(ev.rid)
                        aborted.add(ev.rid)
                # In-process engine: step() runs exactly one scheduler + model
                # step. Step while requests are live, and while vLLM itself says
                # it has work (scheduler.has_requests(): e.g. a connector's
                # pending stores), as vLLM's own engine loop does.
                if live or self._engine_busy():
                    if self.lmcache is not None:
                        self._settle_lmcache()
                    for out in self.engine.step():
                        rid = int(str(out.request_id).split("-")[0])
                        if self.tracing:
                            tag = ("" if rid in prompts and rid not in aborted
                                   else " LEFTOVER" if rid not in prompts else " after-abort")
                            trace.append((f"{step} out {rid} n={len(out.outputs[0].token_ids)}"
                                          f"{' done' if out.finished else ''}{tag}", ""))
                        if rid in aborted or rid not in prompts:
                            continue  # aborted, or a leftover from an earlier run
                        outputs[rid] = list(out.outputs[0].token_ids)
                        if out.finished:
                            live.discard(rid)
                            finished.add(rid)
                if self.tracing:
                    sched = self._scheduler()
                    if sched is not None:
                        from collections import Counter

                        states = Counter(r.status.name for r in sched.waiting)
                        trace.append((f"{step} sched running={len(sched.running)} "
                                      f"waiting={len(sched.waiting)}",
                                      " ".join(f"{k}={v}" for k, v in sorted(states.items()))))
                    if xfer is not None:
                        for tick, phase, kind, event, n, blocks in xfer.TRACE:
                            trace.append((f"{step} {kind} {phase} n={n}",
                                          f"event={event} src->dst={blocks}"))
                        xfer.TRACE.clear()
            except Exception as e:  # vLLM itself failed (e.g. an internal assert)
                import traceback

                frame = traceback.extract_tb(e.__traceback__)[-1]
                where = f"{frame.filename.split('site-packages/')[-1].split('vllm_source/')[-1]}"
                engine_error = (f"step {step}: {type(e).__name__}: {e} "
                                f"at {where}:{frame.lineno} `{(frame.line or '').strip()}`")[:400]
                self.poisoned = f"seed {seed}, {engine_error}"
                break
            step += 1
            if step > self.cfg.max_steps:
                break

        # Offload transfers still in flight: keep the engine stepping (with a
        # keep-alive request) until they complete, so none lands in the next run.
        if self.has_copy_engine and engine_error is None and not live:
            step, engine_error = self._drain_offload(seed, step, prompts, outputs, finished,
                                                     trace if self.tracing else None)

        # Never leave requests behind: they would hold blocks (so the next
        # reset_prefix_cache fails) and emit outputs into the next run.
        leftovers = [str(r) for r in live]
        if leftovers:
            try:
                self.engine.abort_request(leftovers)
            except Exception:
                pass

        if xfer is not None:
            xfer.TRACE_ON = False
        mismatches = []
        for rid, got in sorted(outputs.items()):
            want = expected_continuation(list(prompts[rid]), len(got), self.vocab)
            if got != want:
                mismatches.append(f"req {rid}: first diff at output token "
                                  f"{next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)}")
        after = self.counters()
        metrics = {k.split(":")[1]: after[k] - before[k] for k in self.COUNTERS}
        mafter = self.model_stats()
        metrics["prefill_continuations"] = (mafter.get("prefill_continuations", 0)
                                            - mbefore.get("prefill_continuations", 0))
        metrics["max_step_tokens"] = mafter.get("max_step_tokens", 0)
        oafter = self.offload_stats()
        for k, v in oafter.items():
            metrics["offload_" + k] = v - obefore.get(k, 0)
        if not reset_ok:
            self.resets_refused += 1  # this run was not isolated from the previous one
        return VllmResult(seed, step, len(finished), len(aborted), mismatches,
                          self._violations_since(offset),
                          liveness=bool(live) and engine_error is None, outputs=outputs,
                          metrics=metrics, engine_error=engine_error, trace=trace)


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
