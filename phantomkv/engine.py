"""ToyEngine: a small paged-KV serving engine used to validate Phantom itself.

It mirrors the parts of vLLM / SGLang where isolation bugs live:
  * a paged block pool with reference counts
  * hash-based prefix caching with LRU eviction of unreferenced blocks
  * continuous batching with recompute-style preemption
  * prefill/decode (PD) disaggregation: prompt KV arrives as asynchronous,
    one-sided "RDMA" writes from a remote prefill engine
  * client aborts at any moment

`plant` lets tests switch on known bug classes, so we can prove the oracle
catches them before pointing it at real engines:
  * "abort-free-race"     free a PD request's blocks on abort while KV writes
                          are still in flight (the SGLang Oct-2026 bug class)
  * "stale-prefix-cache"  evict a cached block without dropping its hash entry
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .provenance import ROOT, Tag, chain_step, fold, next_token

if TYPE_CHECKING:
    from .sim import Simulator

PLANTS = ("abort-free-race", "stale-prefix-cache")


@dataclass
class Violation:
    kind: str
    time: int
    req: int
    pos: int
    detail: str

    def __str__(self) -> str:
        return f"[t={self.time}] {self.kind}: req {self.req} pos {self.pos}: {self.detail}"


@dataclass
class Request:
    rid: int
    prompt: list[int]
    max_new: int
    remote: bool = False  # prompt KV comes from a remote prefill engine
    out: list[int] = field(default_factory=list)
    chain: list[int] = field(default_factory=list)
    blocks: list[int] = field(default_factory=list)
    computed: int = 0
    state: str = "waiting"  # waiting | waiting_kv | running | done | aborted
    inflight: int = 0  # remote KV writes not yet landed
    cached_upto: int = 0  # number of leading blocks registered in prefix cache

    def __post_init__(self) -> None:
        h = ROOT
        for t in self.prompt:
            h = chain_step(h, t)
            self.chain.append(h)

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.out

    def append(self, tok: int) -> None:
        self.out.append(tok)
        self.chain.append(chain_step(self.chain[-1], tok))


class ToyEngine:
    def __init__(self, sim: "Simulator", num_blocks: int, block_size: int,
                 max_running: int, plant: str | None = None) -> None:
        if plant is not None and plant not in PLANTS:
            raise ValueError(f"unknown plant {plant!r}; choose from {PLANTS}")
        self.sim, self.B, self.plant = sim, block_size, plant
        self.max_running = max_running
        self.mem: list[Tag | None] = [None] * (num_blocks * block_size)
        self.ref = [0] * num_blocks
        self.free: deque[int] = deque(range(num_blocks))
        self.evictable: OrderedDict[int, int] = OrderedDict()  # block -> key
        self.cache: dict[int, int] = {}  # chain-hash key -> block
        self.block_key: dict[int, int] = {}  # block -> key it's cached under
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.kv_pending: dict[int, Request] = {}
        self.reqs: dict[int, Request] = {}
        self.violations: list[Violation] = []
        self._reported: set[int] = set()

    # ------------------------------------------------------------- blocks
    def _available(self) -> int:
        return len(self.free) + len(self.evictable)

    def _alloc(self) -> int:
        if self.free:
            b = self.free.popleft()
        else:
            b, key = self.evictable.popitem(last=False)  # LRU
            if self.plant != "stale-prefix-cache":
                del self.cache[key]
            del self.block_key[b]
        self.ref[b] = 1
        return b

    def _release(self, blocks: list[int]) -> None:
        for b in blocks:
            self.ref[b] -= 1
            assert self.ref[b] >= 0, "refcount underflow"
            if self.ref[b] == 0:
                if b in self.block_key:
                    self.evictable[b] = self.block_key[b]
                else:
                    self.free.append(b)

    def _slot(self, r: Request, pos: int) -> int:
        return r.blocks[pos // self.B] * self.B + pos % self.B

    def _blocks_for(self, n_tokens: int) -> int:
        return -(-n_tokens // self.B)

    # ------------------------------------------------------------ arrivals
    def submit(self, r: Request) -> None:
        self.reqs[r.rid] = r
        self.waiting.append(r)
        self.sim.log(f"arrive req {r.rid} len={len(r.prompt)} remote={r.remote}")

    def abort(self, rid: int) -> None:
        r = self.reqs.get(rid)
        if r is None or r.state in ("done", "aborted"):
            return
        prev, r.state = r.state, "aborted"
        self.sim.log(f"abort req {rid} (was {prev}, inflight={r.inflight})")
        if prev == "waiting":
            self.waiting.remove(r)
            return
        if prev == "running":
            self.running.remove(r)
        if prev == "waiting_kv":
            self.kv_pending.pop(rid, None)
            if r.inflight > 0 and self.plant != "abort-free-race":
                return  # correct: blocks are freed when the last write lands
        self._release(r.blocks)
        r.blocks = []

    def kv_landed(self, rid: int, block_idx: int, block: int, tags: list[Tag]) -> None:
        """A one-sided remote write lands. Like RDMA, it does not check ownership."""
        for i, tag in enumerate(tags):
            self.mem[block * self.B + i] = tag
        r = self.reqs[rid]
        r.inflight -= 1
        self.sim.log(f"kv-land req {rid} blk#{block_idx}->{block} left={r.inflight}")
        if r.inflight == 0:
            if r.state == "waiting_kv":
                r.computed = len(r.prompt)
                r.state = "running"
                del self.kv_pending[rid]
                self.running.append(r)
            elif r.state == "aborted" and r.blocks:
                self._release(r.blocks)
                r.blocks = []

    # ---------------------------------------------------------- scheduling
    def _admit(self) -> None:
        while self.waiting and len(self.running) + len(self.kv_pending) < self.max_running:
            r = self.waiting[0]
            n = len(r.tokens)
            if r.remote and not r.out:
                if self._available() < self._blocks_for(n):
                    return
                self.waiting.popleft()
                r.blocks = [self._alloc() for _ in range(self._blocks_for(n))]
                r.state, r.inflight = "waiting_kv", len(r.blocks)
                self.kv_pending[r.rid] = r
                self.sim.schedule_remote_prefill(r, self.B)
                continue
            # local: reuse cached full blocks, leaving >=1 token to compute
            hits: list[int] = []
            for i in range((n - 1) // self.B):
                b = self.cache.get(r.chain[(i + 1) * self.B - 1])
                if b is None:
                    break
                hits.append(b)
            need = self._blocks_for(n) - len(hits) + 1
            evictable_hits = sum(1 for b in hits if self.ref[b] == 0)
            if self._available() - evictable_hits < need:
                return
            self.waiting.popleft()
            for b in hits:
                if self.ref[b] == 0:
                    if b in self.evictable:
                        del self.evictable[b]
                    else:  # only reachable when the cache index is already wrong
                        self.free.remove(b)
                self.ref[b] += 1
            r.blocks, r.computed, r.cached_upto = hits, len(hits) * self.B, len(hits)
            r.state = "running"
            self.running.append(r)
            if hits:
                self.sim.log(f"prefix-hit req {r.rid} blocks={len(hits)}")

    def _preempt_for(self, needy: Request) -> bool:
        for victim in reversed(self.running):
            if victim is not needy:
                self.running.remove(victim)
                self._release(victim.blocks)
                victim.blocks, victim.computed, victim.cached_upto = [], 0, 0
                victim.state = "waiting"
                self.waiting.appendleft(victim)
                self.sim.log(f"preempt req {victim.rid} for req {needy.rid}")
                return True
        return False

    def step(self) -> None:
        self._admit()
        for r in list(self.running):
            if r.state != "running":
                continue
            n = len(r.tokens)
            while len(r.blocks) < self._blocks_for(n):
                if self._available() == 0 and not self._preempt_for(r):
                    break
                if self._available() == 0:
                    continue
                r.blocks.append(self._alloc())
            if len(r.blocks) < self._blocks_for(n):
                continue  # stalled this step; try again next tick
            for p in range(r.computed, n):  # write this request's own KV
                self.mem[self._slot(r, p)] = Tag(r.rid, p, r.chain[p])
            r.computed = n
            self._register_full_blocks(r)
            r.append(self._phantom_forward(r, n))
            if len(r.out) >= r.max_new:
                r.state = "done"
                self.running.remove(r)
                self._release(r.blocks)
                r.blocks = []
                self.sim.log(f"finish req {r.rid}")

    def _register_full_blocks(self, r: Request) -> None:
        full = r.computed // self.B
        for i in range(r.cached_upto, full):
            key, b = r.chain[(i + 1) * self.B - 1], r.blocks[i]
            if key not in self.cache and b not in self.block_key:
                self.cache[key] = b
                self.block_key[b] = key
        r.cached_upto = max(r.cached_upto, full)

    # -------------------------------------------------------------- oracle
    def _phantom_forward(self, r: Request, n: int) -> int:
        """Read every KV slot of the context, check provenance, emit a token."""
        acc = ROOT
        for p in range(n):
            tag = self.mem[self._slot(r, p)]
            if tag is None:
                self._violate("unwritten-read", r, p, "slot was never written")
                acc = fold(acc, 0)
                continue
            if tag.chain != r.chain[p]:
                leak = tag.writer != r.rid
                self._violate(
                    "cross-request-leak" if leak else "stale-self-read", r, p,
                    f"read KV of req {tag.writer} pos {tag.pos} (prefix differs)")
            acc = fold(acc, tag.chain)
        return next_token(acc)

    def _violate(self, kind: str, r: Request, pos: int, detail: str) -> None:
        if r.rid in self._reported:
            return  # one report per request keeps output readable
        self._reported.add(r.rid)
        v = Violation(kind, self.sim.now, r.rid, pos, detail)
        self.violations.append(v)
        self.sim.log(f"VIOLATION {v}")

    # ----------------------------------------------------------- liveness
    def idle(self) -> bool:
        return not (self.waiting or self.running or self.kv_pending)

    def check_quiescent(self) -> None:
        """When everything is finished, every block must be reclaimable."""
        leaked = [b for b, c in enumerate(self.ref) if c != 0]
        if leaked:
            self.violations.append(Violation(
                "block-leak", self.sim.now, -1, -1, f"blocks still referenced: {leaked[:8]}"))
