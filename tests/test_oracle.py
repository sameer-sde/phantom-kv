"""The oracle must (1) never cry wolf on a correct engine and (2) catch every
planted bug class. If either fails, nothing Phantom reports can be trusted."""

import pytest

from phantomkv import PLANTS, Config, run_seed
from phantomkv.provenance import chain_of

SEEDS = range(300)


def test_chain_hash_identifies_prefix():
    a, b = chain_of([1, 2, 3]), chain_of([1, 2, 4])
    assert a[:2] == b[:2] and a[2] != b[2]


def test_no_false_positives_on_correct_engine():
    for seed in SEEDS:
        res = run_seed(seed)
        assert res.ok, f"seed {seed}: {res.violations[0]}"
        assert res.finished + res.aborted == Config().num_requests


def test_no_false_positives_under_heavy_pressure():
    cfg = Config(num_blocks=12, max_running=8, abort_frac=0.5, remote_frac=0.8)
    for seed in SEEDS:
        res = run_seed(seed, cfg)
        assert res.ok, f"seed {seed}: {res.violations[0]}"


def test_prefix_cache_is_exercised():
    hits = sum("prefix-hit" in line for s in range(20) for line in run_seed(s).trace)
    assert hits > 0, "workload never hits the prefix cache; test would be vacuous"


@pytest.mark.parametrize("plant", PLANTS)
def test_planted_bug_is_caught(plant):
    caught = [s for s in SEEDS if not run_seed(s, plant=plant).ok]
    assert caught, f"{plant} was never detected"


def test_abort_race_is_reported_as_cross_request_leak():
    res = next(r for s in SEEDS if not (r := run_seed(s, plant="abort-free-race")).ok)
    assert res.violations[0].kind == "cross-request-leak"


def test_same_seed_same_run():
    for seed in (0, 7, 123):
        assert run_seed(seed).fingerprint == run_seed(seed).fingerprint
        assert (run_seed(seed, plant="abort-free-race").fingerprint
                == run_seed(seed, plant="abort-free-race").fingerprint)


def test_different_seeds_differ():
    assert len({run_seed(s).fingerprint for s in range(10)}) == 10
