"""Deterministic simulator (FoundationDB / TigerBeetle style).

A run is fully described by (seed, events):
  * `generate_workload(seed, cfg)` turns a seed into an explicit list of
    Events (arrivals with prompt contents, client aborts);
  * the simulator replays that list on a virtual clock;
  * the only other randomness, KV-transfer network delay, is a pure hash of
    (seed, request id, block index) -- NOT a sequential PRNG draw.

That last point matters for shrinking: removing an event must not shift the
random choices made for every event after it, or the bug would vanish for
the wrong reason. Same (seed, events) -> same run, bit for bit.
"""

from __future__ import annotations

import hashlib
import heapq
import random
from dataclasses import dataclass, field
from typing import Callable

from .engine import Request, ToyEngine, Violation
from .provenance import Tag, _h


@dataclass
class Config:
    num_requests: int = 40
    num_blocks: int = 24
    block_size: int = 4
    max_running: int = 6
    max_prompt: int = 24
    max_new: int = 10
    shared_prefixes: int = 3  # system prompts shared across requests
    remote_frac: float = 0.5  # fraction of requests using PD disaggregation
    abort_frac: float = 0.25
    max_net_delay: int = 12  # ticks a remote KV write may take to land
    arrival_window: int = 60
    max_ticks: int = 5_000


@dataclass(frozen=True)
class Event:
    """One input to the simulation."""
    time: int
    kind: str  # "submit" | "abort"
    rid: int
    prompt: tuple[int, ...] = ()  # tuple: frozen dataclasses must be hashable
    max_new: int = 0
    remote: bool = False

    def __str__(self) -> str:
        if self.kind == "abort":
            return f"t={self.time:<4} abort  req {self.rid}"
        where = "remote-prefill" if self.remote else "local"
        return (f"t={self.time:<4} submit req {self.rid} "
                f"prompt_len={len(self.prompt)} max_new={self.max_new} {where}")


def generate_workload(seed: int, cfg: Config) -> list[Event]:
    """Seed -> explicit, editable list of events."""
    c, rng = cfg, random.Random(seed)
    max_tokens = (c.num_blocks - 1) * c.block_size  # any single request fits
    prefixes = [[rng.randrange(1, 1000) for _ in range(rng.randrange(4, 13))]
                for _ in range(c.shared_prefixes)]
    events: list[Event] = []
    for rid in range(c.num_requests):
        body = [rng.randrange(1, 1000) for _ in range(rng.randrange(1, c.max_prompt))]
        prompt = (rng.choice(prefixes) + body) if rng.random() < 0.6 else body
        max_new = 1 + rng.randrange(c.max_new)
        prompt = prompt[: max(1, max_tokens - max_new)]
        remote = rng.random() < c.remote_frac
        t = rng.randrange(c.arrival_window)
        events.append(Event(t, "submit", rid, tuple(prompt), max_new, remote))
        if rng.random() < c.abort_frac:
            events.append(Event(t + rng.randrange(1, 25), "abort", rid))
    return events


@dataclass
class Result:
    seed: int
    violations: list[Violation]
    ticks: int
    trace: list[str] = field(repr=False)
    finished: int = 0
    aborted: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256("\n".join(self.trace).encode()).hexdigest()[:16]


class Simulator:
    def __init__(self, seed: int, cfg: Config, plant: str | None = None,
                 events: list[Event] | None = None) -> None:
        self.seed, self.cfg = seed, cfg
        self.events = events if events is not None else generate_workload(seed, cfg)
        self.now = 0
        self._q: list[tuple[int, int, Callable[[], None]]] = []
        self._seq = 0
        self.trace: list[str] = []
        self.engine = ToyEngine(self, cfg.num_blocks, cfg.block_size,
                                cfg.max_running, plant)

    # ------------------------------------------------------------ plumbing
    def log(self, msg: str) -> None:
        self.trace.append(f"t={self.now:<5} {msg}")

    def at(self, t: int, fn: Callable[[], None]) -> None:
        heapq.heappush(self._q, (t, self._seq, fn))
        self._seq += 1

    def net_delay(self, rid: int, block_idx: int) -> int:
        """Stable per-(request, block) delay: independent of other events."""
        return 1 + _h(self.seed, rid, block_idx) % self.cfg.max_net_delay

    def schedule_remote_prefill(self, r: Request, B: int) -> None:
        """Remote prefill engine computes the prompt KV (correct by
        construction) and streams it block by block with network delay."""
        for i, blk in enumerate(r.blocks):
            lo, hi = i * B, min((i + 1) * B, len(r.prompt))
            tags = [Tag(r.rid, p, r.chain[p]) for p in range(lo, hi)]
            self.at(self.now + self.net_delay(r.rid, i),
                    lambda rid=r.rid, i=i, blk=blk, tags=tags:
                    self.engine.kv_landed(rid, i, blk, tags))

    # ------------------------------------------------------------ workload
    def _schedule_events(self) -> None:
        for ev in self.events:
            if ev.kind == "submit":
                r = Request(ev.rid, list(ev.prompt), ev.max_new, ev.remote)
                self.at(ev.time, lambda r=r: self.engine.submit(r))
            elif ev.kind == "abort":
                self.at(ev.time, lambda rid=ev.rid: self.engine.abort(rid))
            else:
                raise ValueError(f"unknown event kind {ev.kind!r}")

    # ---------------------------------------------------------------- run
    def run(self) -> Result:
        self._schedule_events()
        e = self.engine
        while self.now < self.cfg.max_ticks:
            while self._q and self._q[0][0] <= self.now:
                heapq.heappop(self._q)[2]()
            e.step()
            if not self._q and e.idle():
                break
            self.now += 1
        if self.now >= self.cfg.max_ticks:
            stuck = [r.rid for r in e.reqs.values() if r.state not in ("done", "aborted")]
            e.violations.append(Violation("liveness", self.now, -1, -1,
                                          f"requests never finished: {stuck[:8]}"))
        else:
            e.check_quiescent()
        states = [r.state for r in e.reqs.values()]
        return Result(self.seed, e.violations, self.now, self.trace,
                      states.count("done"), states.count("aborted"))


def run_events(seed: int, events: list[Event], cfg: Config | None = None,
               plant: str | None = None) -> Result:
    return Simulator(seed, cfg or Config(), plant, events).run()


def run_seed(seed: int, cfg: Config | None = None, plant: str | None = None) -> Result:
    return Simulator(seed, cfg or Config(), plant).run()
