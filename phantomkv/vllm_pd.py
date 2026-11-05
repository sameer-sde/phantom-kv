"""Seeded fuzzing of disaggregated prefill/decode (PD) on real vLLM.

Day 14 checked one hand-written scenario (abort half the requests while
their KV is in flight). This module turns that into seeded fuzzing:

    seed -> PdScenario(delay, events)

  * a POOL of prompts is prefilled once by a separate prefill engine, whose
    connector saves their KV to shared storage (two processes, like two
    machines); the pool is reused across seeds and campaigns
  * every seed picks a load delay (KV lands 1-5 engine steps after the load
    starts), which pool prompts to transfer and when, which of them to abort
    and when (before the load starts, while it is in flight, or after it
    lands), and when local "filler" requests arrive to take free blocks
  * one decode engine (in-process, so add/abort/step happen in strict order)
    runs seed after seed; it loads transferred KV asynchronously through
    PhantomAsyncFaultConnector and never saves, so replays are exact

Phantom checks every slot the decode engine reads, and every output token is
compared with the pure-Python reference. Failing seeds shrink with ddmin
(phantomkv pd-shrink), the delay held fixed.

The workload generator is pure Python (tested without vLLM); PdRunner and
the prefill step need the vLLM environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

from .sim import Event
from .tags import expected_continuation
from .vllm_harness import VllmResult, page_bytes, primary_kind

POOL_SEED = 0
KEEPALIVE_RID = 10_000  # rids >= this are harness keep-alive requests, not events


@dataclass
class PdConfig:
    pool_size: int = 16
    transfers: int = 8  # transferred requests per seed (distinct pool prompts)
    fillers: int = 8  # locally computed requests per seed
    arrival_window: int = 12  # engine steps
    abort_frac: float = 0.5  # of transferred requests; fillers get half of it
    delays: tuple[int, ...] = (1, 2, 3, 4, 5)  # async load delay, engine steps
    max_new: int = 12
    block_size: int = 32
    kv_blocks: int = 96
    max_num_seqs: int = 8
    max_model_len: int = 512
    max_steps: int = 1_000


@dataclass
class PdScenario:
    seed: int
    delay: int
    events: list[Event]


def pd_pool(n: int, seed: int = POOL_SEED) -> list[list[int]]:
    """The prompts the prefill engine computes once. Half share a 2-block
    prefix at the start, half carry part of it at the end."""
    rng = random.Random(seed)
    shared = [rng.randrange(1, 50_000) for _ in range(64)]
    out = []
    for i in range(n):
        body = [rng.randrange(1, 50_000) for _ in range(rng.randrange(10, 140))]
        out.append(shared + body if i % 2 == 0 else body + shared[:20])
    return out


def pd_workload(seed: int, cfg: PdConfig) -> PdScenario:
    rng = random.Random(seed)
    delay = rng.choice(cfg.delays)
    pool = pd_pool(cfg.pool_size)
    events: list[Event] = []
    rid = 0
    for idx in rng.sample(range(cfg.pool_size), min(cfg.transfers, cfg.pool_size)):
        t = rng.randrange(cfg.arrival_window)
        events.append(Event(t, "submit", rid, tuple(pool[idx]), 1 + rng.randrange(cfg.max_new),
                            remote=True))
        if rng.random() < cfg.abort_frac:
            # 0 = before the load starts; 1..delay = in flight; later = after landing
            events.append(Event(t + rng.randrange(0, delay + 3), "abort", rid))
        rid += 1
    for _ in range(cfg.fillers):
        body = [rng.randrange(1, 50_000) for _ in range(rng.randrange(20, 140))]
        # arrive throughout, including just after aborts, to take freed blocks
        t = rng.randrange(cfg.arrival_window + delay + 4)
        events.append(Event(t, "submit", rid, tuple(body), 1 + rng.randrange(cfg.max_new)))
        if rng.random() < cfg.abort_frac / 2:
            events.append(Event(t + rng.randrange(1, 10), "abort", rid))
        rid += 1
    return PdScenario(seed, delay, events)


# ------------------------------------------------------------------ vLLM side
def _quiet_connector_logs() -> None:
    """ExampleConnector logs a line per load and warns on every step without a
    forward pass; across a campaign that buries the results. Errors still show."""
    import logging

    log = logging.getLogger("vllm.distributed.kv_transfer.kv_connector.v1.example_connector")
    log.setLevel(logging.ERROR)
    # a filter too: logging.config.dictConfig resets child loggers' levels but
    # keeps their filters
    if not any(isinstance(f, _BelowError) for f in log.filters):
        log.addFilter(_BelowError())


class _BelowError:
    def filter(self, record) -> bool:  # logging.Filter protocol
        import logging

        return record.levelno >= logging.ERROR


def build_pd_llm(cfg: PdConfig, storage: str, asynchronous: bool):
    from vllm import LLM
    from vllm.config import KVTransferConfig

    _quiet_connector_logs()
    llm = LLM(
        model="facebook/opt-125m",
        hf_overrides={"architectures": ["PhantomForCausalLM"]},
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="float32",
        enforce_eager=True,
        # Off on purpose: any context the decode engine did not compute itself
        # can only have arrived through the connector.
        enable_prefix_caching=False,
        block_size=cfg.block_size,
        max_model_len=cfg.max_model_len,
        max_num_seqs=cfg.max_num_seqs,
        kv_cache_memory_bytes=cfg.kv_blocks * page_bytes(cfg.block_size),
        seed=0,
        kv_transfer_config=KVTransferConfig(
            kv_connector="PhantomAsyncFaultConnector" if asynchronous else "PhantomFaultConnector",
            kv_connector_module_path="phantomkv.vllm_fault_connector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": storage},
        ),
    )
    # after the engine is built: vLLM (re)configures its loggers during start-up,
    # which resets levels set earlier
    _quiet_connector_logs()
    return llm


def _pool_marker(cfg: PdConfig) -> dict:
    blob = json.dumps(pd_pool(cfg.pool_size)).encode()
    return {"pool_size": cfg.pool_size, "block_size": cfg.block_size,
            "pool_sha": hashlib.sha256(blob).hexdigest()[:16]}


def ensure_pool(cfg: PdConfig, storage: str) -> bool:
    """Prefill the pool into `storage` unless it is already there.
    Runs the prefill engine in a separate process. Returns True if it ran."""
    marker = os.path.join(storage, "phantom_pool.json")
    want = _pool_marker(cfg)
    try:
        if json.load(open(marker)) == want:
            return False
    except (OSError, ValueError):
        pass
    shutil.rmtree(storage, ignore_errors=True)
    os.makedirs(storage)
    env = dict(os.environ)
    for k in ("PHANTOM_PD_FAULT", "PHANTOM_VLLM_PLANT", "PHANTOM_PD_ASYNC_DELAY",
              "PHANTOM_PD_NO_SAVE", "PHANTOM_PD_FAULT_LOG"):
        env.pop(k, None)
    env["PHANTOM_REPORT"] = os.path.abspath("phantom_report_prefill.jsonl")
    code = subprocess.call([sys.executable, "-m", "phantomkv", "pd-prefill",
                            "--storage", storage, "--pool", str(cfg.pool_size)], env=env)
    if code != 0:
        raise RuntimeError(f"prefill engine failed (exit {code})")
    with open(marker, "w") as f:
        json.dump(want, f)
    return True


def run_prefill(cfg: PdConfig, storage: str) -> int:
    """Inside the prefill process: compute the pool, the connector saves its KV."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    report = os.environ["PHANTOM_REPORT"]
    for p in (report, report + ".stats.json"):
        if os.path.exists(p):
            os.remove(p)
    llm = build_pd_llm(cfg, storage, asynchronous=False)
    vocab = llm.llm_engine.model_config.hf_config.vocab_size
    pool = pd_pool(cfg.pool_size)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pool],
                        SamplingParams(max_tokens=1, temperature=0, detokenize=False),
                        use_tqdm=False)
    bad = sum(list(o.outputs[0].token_ids) != expected_continuation(p, 1, vocab)
              for p, o in zip(pool, outs))
    violations = sum(1 for _ in open(report)) if os.path.exists(report) else 0
    saved = sum(len(fs) for _, _, fs in os.walk(storage))
    print(f"prefill: {len(pool)} pool prompts, KV files saved: {saved}, "
          f"wrong first tokens: {bad}, violations: {violations}")
    return 0 if (bad == 0 and violations == 0 and saved) else 1


class PdRunner:
    """One decode engine, reused across seeds (building it takes seconds)."""

    def __init__(self, cfg: PdConfig | None = None, storage: str = "pd_fuzz_storage",
                 fault: str | None = None, plant: str | None = None,
                 report: str = "phantom_pd_report.jsonl",
                 fault_log: str = "pd_fuzz_events.jsonl") -> None:
        self.cfg = cfg or PdConfig()
        self.storage = os.path.abspath(storage)
        self.report = os.path.abspath(report)
        self.fault_log = os.path.abspath(fault_log)
        self.prefilled = ensure_pool(self.cfg, self.storage)
        for path in (self.report, self.report + ".stats.json", self.fault_log):
            if os.path.exists(path):
                os.remove(path)
        self.poisoned: str | None = None
        self.fault, self.plant = fault, plant
        self.runs = 0
        os.environ.update(PHANTOM_REPORT=self.report, PHANTOM_PD_FAULT_LOG=self.fault_log,
                          PHANTOM_PD_NO_SAVE="1", PHANTOM_PD_ASYNC_DELAY="3",
                          VLLM_ENABLE_V1_MULTIPROCESSING="0")  # exact step ordering
        for var, val in (("PHANTOM_PD_FAULT", fault), ("PHANTOM_VLLM_PLANT", plant)):
            if val:
                os.environ[var] = val
            else:
                os.environ.pop(var, None)
        self.llm = build_pd_llm(self.cfg, self.storage, asynchronous=True)
        self.engine = self.llm.llm_engine
        self.vocab = self.engine.model_config.hf_config.vocab_size

    # ---------------------------------------------------------------- helpers
    def _size(self, path: str) -> int:
        return os.path.getsize(path) if os.path.exists(path) else 0

    def _read_from(self, path: str, offset: int) -> tuple[list[dict], int]:
        if not os.path.exists(path):
            return [], offset
        with open(path) as f:
            f.seek(offset)
            data = f.read()
        return [json.loads(l) for l in data.splitlines() if l.strip()], offset + len(data.encode())

    # -------------------------------------------------------------------- run
    def run(self, seed: int, scenario: PdScenario | None = None) -> VllmResult:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        scen = scenario or pd_workload(seed, self.cfg)
        os.environ["PHANTOM_PD_ASYNC_DELAY"] = str(scen.delay)
        self.runs += 1
        tag = f"r{self.runs}x"  # unique per run; no '-' (vLLM may append one)

        def rid_of(req: str) -> int | None:
            name = str(req).split("-")[0]
            return int(name[len(tag):]) if name.startswith(tag) else None

        offset = self._size(self.report)
        log_off = self._size(self.fault_log)
        by_step: dict[int, list[Event]] = {}
        for ev in scen.events:
            by_step.setdefault(ev.time, []).append(ev)
        last_event_step = max(by_step, default=-1)

        prompts: dict[int, list[int]] = {}
        outputs: dict[int, list[int]] = {}
        live: set[int] = set()
        finished: set[int] = set()
        aborted_at: dict[int, int] = {}
        started_at: dict[int, int] = {}
        landed_at: dict[int, int] = {}
        landed_tokens: dict[int, int] = {}
        keepalives = 0
        step = 0
        engine_error: str | None = None

        def submit(rid: int, prompt: list[int], max_new: int) -> None:
            params = SamplingParams(max_tokens=max_new, temperature=0, detokenize=False)
            self.engine.add_request(f"{tag}{rid}", TokensPrompt(prompt_token_ids=prompt), params)
            prompts[rid] = prompt
            live.add(rid)

        def pending() -> set[int]:
            return set(started_at) - set(landed_at)

        while step <= last_event_step or live or pending():
            try:
                evs = by_step.get(step, ())
                for ev in evs:  # submits first, then aborts (an abort at the
                    if ev.kind == "submit":  # submit step = before the load starts)
                        prompt = list(ev.prompt)
                        if ev.remote:  # the decode side sees prompt + prefill's token
                            prompt += expected_continuation(prompt, 1, self.vocab)
                        submit(ev.rid, prompt, ev.max_new)
                for ev in evs:
                    if ev.kind == "abort" and ev.rid in live:
                        self.engine.abort_request([f"{tag}{ev.rid}"])
                        live.discard(ev.rid)
                        aborted_at[ev.rid] = step
                if not live and step > last_event_step and pending():
                    # KV still in flight for aborted requests: keep the engine
                    # stepping until it lands, so nothing leaks into the next run
                    if keepalives >= 3:
                        break
                    rng = random.Random(seed * 1_000 + keepalives)
                    submit(KEEPALIVE_RID + keepalives,
                           [rng.randrange(1, 50_000) for _ in range(40)], scen.delay + 3)
                    keepalives += 1
                if live:
                    for out in self.engine.step():
                        rid = rid_of(out.request_id)
                        if rid is None or rid in aborted_at:
                            continue
                        outputs[rid] = list(out.outputs[0].token_ids)
                        if out.finished:
                            live.discard(rid)
                            finished.add(rid)
                events, log_off = self._read_from(self.fault_log, log_off)
                for e in events:
                    rid = rid_of(e.get("req", ""))
                    if rid is None:
                        continue
                    if e.get("event") == "started":
                        started_at.setdefault(rid, step)
                    elif e.get("event") == "landed":
                        landed_at.setdefault(rid, step)
                        landed_tokens[rid] = int(e.get("tokens", 0))
            except Exception as e:  # vLLM itself failed (e.g. an internal assert)
                import traceback

                frame = traceback.extract_tb(e.__traceback__)[-1]
                where = frame.filename.split("site-packages/")[-1]
                engine_error = (f"step {step}: {type(e).__name__}: {e} "
                                f"at {where}:{frame.lineno} `{(frame.line or '').strip()}`")[:400]
                self.poisoned = f"seed {seed}, {engine_error}"
                break
            step += 1
            if step > self.cfg.max_steps:
                break

        leftovers = [f"{tag}{r}" for r in live]
        if leftovers:
            try:
                self.engine.abort_request(leftovers)
            except Exception:
                pass

        mismatches = []
        for rid, got in sorted(outputs.items()):
            want = expected_continuation(prompts[rid], len(got), self.vocab)
            if got != want:
                i = next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)
                who = "keep-alive" if rid >= KEEPALIVE_RID else "req"
                mismatches.append(f"{who} {rid}: first diff at output token {i}")
        # an abort was "in flight" if its load had started and had not landed yet
        in_flight = sorted(r for r, a in aborted_at.items()
                           if r in started_at and started_at[r] < a <= landed_at.get(r, 1 << 30))
        metrics = {
            "delay": scen.delay,
            "loads_started": len(started_at),
            "loads_landed": len(landed_at),
            "aborts": len(aborted_at),
            "aborts_in_flight": len(in_flight),
            # KV tokens that landed for requests that were NOT aborted, i.e.
            # transferred context the decode engine then read and Phantom checked
            "transferred_tokens": sum(n for r, n in landed_tokens.items() if r not in aborted_at),
            "keepalives": keepalives,
        }
        stuck = bool(live) or bool(pending())
        return VllmResult(seed, step, len(finished), len(aborted_at), mismatches,
                          self._violations_since(offset),
                          liveness=stuck and engine_error is None, outputs=outputs,
                          metrics=metrics, engine_error=engine_error)

    def _violations_since(self, offset: int) -> list[dict]:
        return self._read_from(self.report, offset)[0]


# ------------------------------------------------------------------ shrinking
@dataclass
class PdShrinkResult:
    seed: int
    delay: int
    kind: str
    original: list[Event]
    minimal: list[Event]
    result: VllmResult
    runs: int
    reproduced: int
    confirm: int
    first: VllmResult = field(repr=False, default=None)  # type: ignore[assignment]


def pd_shrink(runner: PdRunner, seed: int, confirm: int = 3) -> PdShrinkResult | None:
    """ddmin over the seed's events (the load delay stays fixed), re-running
    the decode engine for every candidate; then re-run the minimal list
    `confirm` times to check it reproduces."""
    from .shrink import ddmin

    scen = pd_workload(seed, runner.cfg)
    first = runner.run(seed, scen)
    kind = primary_kind(first)
    if kind is None:
        return None
    runs = 1

    def fails(evs: list[Event]) -> bool:
        nonlocal runs
        runs += 1
        return kind in runner.run(seed, PdScenario(seed, scen.delay, evs)).kinds

    minimal = ddmin(scen.events, fails)
    finals = [runner.run(seed, PdScenario(seed, scen.delay, minimal)) for _ in range(confirm)]
    reproduced = sum(kind in r.kinds for r in finals)
    return PdShrinkResult(seed, scen.delay, kind, scen.events, minimal, finals[0], runs,
                          reproduced, confirm, first)
