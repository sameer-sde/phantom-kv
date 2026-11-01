"""Shrinking must produce small, still-failing, 1-minimal reproductions."""

import pytest

from phantomkv import PLANTS
from phantomkv.shrink import ddmin, shrink_seed
from phantomkv.sim import Config, Simulator, generate_workload, run_events, run_seed


def test_ddmin_finds_minimal_pair():
    assert ddmin(list(range(50)), lambda xs: 3 in xs and 41 in xs) == [3, 41]


def test_ddmin_rejects_passing_input():
    with pytest.raises(ValueError):
        ddmin([1, 2, 3], lambda xs: False)


def test_event_list_reproduces_seed_exactly():
    for seed in (0, 5, 42):
        events = generate_workload(seed, Config())
        assert run_events(seed, events).fingerprint == run_seed(seed).fingerprint


def test_network_delay_independent_of_other_events():
    """Dropping events must not change anyone else's network delays,
    otherwise shrinking would perturb the very interleaving it is isolating."""
    cfg = Config()
    events = generate_workload(3, cfg)
    full = Simulator(3, cfg, events=events)
    fewer = Simulator(3, cfg, events=events[::3])
    for rid in range(cfg.num_requests):
        for blk in range(cfg.num_blocks):
            assert full.net_delay(rid, blk) == fewer.net_delay(rid, blk)
    assert len({full.net_delay(1, b) for b in range(cfg.num_blocks)}) > 1  # not constant


def _first_failing(plant):
    return next(s for s in range(200) if not run_seed(s, plant=plant).ok)


@pytest.mark.parametrize("plant", PLANTS)
def test_shrunk_repro_is_small_and_still_fails(plant):
    res = shrink_seed(_first_failing(plant), plant=plant)
    assert res is not None
    assert len(res.minimal) <= 15
    assert len(res.minimal) < len(res.original)
    assert any(v.kind == res.violation.kind for v in res.result.violations)


@pytest.mark.parametrize("plant", PLANTS)
def test_shrunk_repro_is_one_minimal(plant):
    seed = _first_failing(plant)
    res = shrink_seed(seed, plant=plant)
    kind = res.violation.kind
    for i in range(len(res.minimal)):
        smaller = res.minimal[:i] + res.minimal[i + 1:]
        out = run_events(seed, smaller, plant=plant)
        assert not any(v.kind == kind for v in out.violations), \
            f"event {i} was removable: {res.minimal[i]}"


def test_passing_seed_returns_none():
    assert shrink_seed(0) is None
