"""Seeded PD workloads (phantomkv/vllm_pd.py), checked without vLLM."""

from phantomkv.vllm_pd import PdConfig, pd_pool, pd_workload

CFG = PdConfig()


def test_same_seed_same_scenario():
    assert pd_workload(7, CFG) == pd_workload(7, CFG)
    assert pd_workload(7, CFG) != pd_workload(8, CFG)


def test_pool_is_fixed_and_transfers_come_from_it():
    assert pd_pool(16) == pd_pool(16)
    pool = {tuple(p) for p in pd_pool(CFG.pool_size)}
    for seed in range(50):
        sc = pd_workload(seed, CFG)
        remote = [e for e in sc.events if e.kind == "submit" and e.remote]
        local = [e for e in sc.events if e.kind == "submit" and not e.remote]
        assert len(remote) == CFG.transfers and len(local) == CFG.fillers
        assert all(e.prompt in pool for e in remote)
        assert len({e.prompt for e in remote}) == len(remote)  # distinct pool prompts
        assert not any(e.prompt in pool for e in local)  # fillers never hit storage


def test_delays_and_abort_timings_cover_every_phase():
    """Across seeds: every delay, and aborts before the load starts (same
    step as submit), while it is in flight, and after it has landed."""
    delays, phases = set(), set()
    for seed in range(200):
        sc = pd_workload(seed, CFG)
        delays.add(sc.delay)
        submit_at = {e.rid: e.time for e in sc.events if e.kind == "submit" and e.remote}
        for e in sc.events:
            if e.kind == "abort" and e.rid in submit_at:
                lag = e.time - submit_at[e.rid]
                phases.add("before" if lag == 0 else "in-flight" if lag <= sc.delay else "after")
    assert delays == set(CFG.delays)
    assert phases == {"before", "in-flight", "after"}


def test_prompts_fit_the_model():
    for seed in range(100):
        for e in pd_workload(seed, CFG).events:
            if e.kind == "submit":
                assert len(e.prompt) + 1 + e.max_new < CFG.max_model_len
                assert e.max_new >= 1
