"""Deterministic simulator (FoundationDB / TigerBeetle style).

Every source of nondeterminism -- arrivals, prompt contents, aborts, network
delay of KV transfers -- is drawn from ONE seeded PRNG. Same seed, same run,
bit for bit. A failing seed is therefore a complete bug report.
"""

from __future__ import annotations

import hashlib
import heapq
import random
from dataclasses import dataclass, field
from typing import Callable

from .engine import Request, ToyEngine, Violation
from .provenance import Tag


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
    def __init__(self, seed: int, cfg: Config, plant: str | None = None) -> None:
        self.seed, self.cfg = seed, cfg
        self.rng = random.Random(seed)
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

    def schedule_remote_prefill(self, r: Request, B: int) -> None:
        """Remote prefill engine computes the prompt KV (correct by
        construction) and streams it block by block with random delays."""
        for i, blk in enumerate(r.blocks):
            lo, hi = i * B, min((i + 1) * B, len(r.prompt))
            tags = [Tag(r.rid, p, r.chain[p]) for p in range(lo, hi)]
            delay = 1 + self.rng.randrange(self.cfg.max_net_delay)
            self.at(self.now + delay,
                    lambda rid=r.rid, i=i, blk=blk, tags=tags:
                    self.engine.kv_landed(rid, i, blk, tags))

    # ------------------------------------------------------------ workload
    def _workload(self) -> None:
        c, rng = self.cfg, self.rng
        max_tokens = (c.num_blocks - 1) * c.block_size  # always fits alone
        prefixes = [[rng.randrange(1, 1000) for _ in range(rng.randrange(4, 13))]
                    for _ in range(c.shared_prefixes)]
        for rid in range(c.num_requests):
            body = [rng.randrange(1, 1000) for _ in range(rng.randrange(1, c.max_prompt))]
            prompt = (rng.choice(prefixes) + body) if rng.random() < 0.6 else body
            max_new = 1 + rng.randrange(c.max_new)
            prompt = prompt[: max(1, max_tokens - max_new)]
            r = Request(rid, prompt, max_new, remote=rng.random() < c.remote_frac)
            t = rng.randrange(c.arrival_window)
            self.at(t, lambda r=r: self.engine.submit(r))
            if rng.random() < c.abort_frac:
                self.at(t + rng.randrange(1, 25), lambda rid=rid: self.engine.abort(rid))

    # ---------------------------------------------------------------- run
    def run(self) -> Result:
        self._workload()
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


def run_seed(seed: int, cfg: Config | None = None, plant: str | None = None) -> Result:
    return Simulator(seed, cfg or Config(), plant).run()
