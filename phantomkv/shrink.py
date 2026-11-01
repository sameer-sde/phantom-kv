"""Test-case shrinking: failing run -> minimal reproduction.

Uses delta debugging (Zeller & Hildebrandt, "Simplifying and Isolating
Failure-Inducing Input", IEEE TSE 2002). Given a list of events for which
`fails(events)` is True, ddmin returns a sub-list that still fails and is
1-minimal: removing any single remaining event makes the failure disappear.

What "still fails" means matters: we require the SAME violation kind as the
original failure, so shrinking can't wander off to a different, unrelated bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

from .engine import Violation
from .sim import Config, Event, Result, generate_workload, run_events

T = TypeVar("T")


def ddmin(items: Sequence[T], fails: Callable[[list[T]], bool]) -> list[T]:
    """Return a 1-minimal sub-sequence of `items` for which `fails` holds."""
    items = list(items)
    if not fails(items):
        raise ValueError("ddmin needs a failing input to start from")
    n = 2  # granularity: number of chunks we split into
    while len(items) >= 2:
        size = -(-len(items) // n)  # ceil division
        chunks = [items[i:i + size] for i in range(0, len(items), size)]
        reduced = False
        # 1) try a single chunk on its own
        for c in chunks:
            if len(chunks) > 1 and fails(c):
                items, n, reduced = c, 2, True
                break
        # 2) try removing one chunk (keep the complement)
        if not reduced:
            for i in range(len(chunks)):
                rest = [x for j, c in enumerate(chunks) if j != i for x in c]
                if fails(rest):
                    items, n, reduced = rest, max(n - 1, 2), True
                    break
        if not reduced:
            if n >= len(items):
                break  # already at single-element granularity: 1-minimal
            n = min(n * 2, len(items))
    return items


@dataclass
class ShrinkResult:
    seed: int
    plant: str | None
    original: list[Event]
    minimal: list[Event]
    violation: Violation
    result: Result
    runs: int  # how many simulations the shrink needed

    @property
    def ratio(self) -> float:
        return len(self.minimal) / len(self.original)


def shrink_seed(seed: int, cfg: Config | None = None,
                plant: str | None = None) -> ShrinkResult | None:
    """Shrink the failing run for `seed`. Returns None if the seed passes."""
    cfg = cfg or Config()
    events = generate_workload(seed, cfg)
    first = run_events(seed, events, cfg, plant)
    if first.ok:
        return None
    kind = first.violations[0].kind
    runs = 1

    def fails(evs: list[Event]) -> bool:
        nonlocal runs
        runs += 1
        res = run_events(seed, evs, cfg, plant)
        return any(v.kind == kind for v in res.violations)

    minimal = ddmin(events, fails)
    final = run_events(seed, minimal, cfg, plant)
    violation = next(v for v in final.violations if v.kind == kind)
    return ShrinkResult(seed, plant, events, minimal, violation, final, runs)
