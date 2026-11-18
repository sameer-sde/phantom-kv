# Roadmap

Goal: confirmed bugs in production LLM-serving engines, found by Phantom.

| Phase | Weeks | Deliverable | Done when |
|---|---|---|---|
| 0. Oracle + simulator | 1 | Toy engine, provenance oracle, seeded sim, replay | ✅ 0 false positives / 1k seeds; 2 plants caught |
| 1. Shrinking | 2 | Delta-debugging: failing seed → minimal event trace | ✅ ~50 events → median 6, max 13; 1-minimal (tested) |
| 2. Phantom in vLLM | 3–5 | Out-of-tree vLLM model on CPU backend writing tags through real block tables / slot mapping | ✅ v0.3.0: unmodified vLLM passes clean seeds; a runtime-planted stale prefix-cache bug is caught (15/20 seeds) |
| 3. Drive real vLLM deterministically | 6–8 | Seeded workload + abort/preempt/pressure injection against vLLM `EngineCore` | ✅ v0.4.0: same seed → same trace on real vLLM, across chunked prefill, preemption and block-size profiles |
| 4. Disaggregation + faults | 9–11 | Two engines + simulated KV-transfer layer (delay, reorder, duplicate, abort mid-transfer, handshake failure) | ✅ v0.5.0: SGLang Oct-2026 slot race reproduced and caught; delay, swap, silent-drop, duplicate, abort mid-transfer, all under seeded fuzzing. Reorder, handshake failure and the NIXL class moved to Phase 5 (they need the NIXL connector) |
| 5. Bug hunt | 12–14 | Campaigns on vLLM main, SGLang, LMCache, llm-d | ✅ v0.6.0: vLLM's two CPU offloading connectors (clean, planted faults caught) and LMCache: two real open bugs found independently and shrunk (issues #4642, #5656; already reported, so no duplicates filed); A/B tests show both open fixes work (PRs #6106, #6540). SGLang, llm-d and NIXL not covered (need GPUs) |
| 6. Publish | 15–16 | Paper (arXiv/Zenodo), blog, meetup talk | ✅ v0.7.0: paper (PDF in the repository; not yet submitted to arXiv/Zenodo), blog post, PyPI package with Trusted Publishing. No meetup talk yet |

Highest-yield targets (newest, least-tested paths): push-mode KV transfer,
CPU offloading × quantized KV, hybrid Mamba/attention KV groups, LoRA + prefix cache.
