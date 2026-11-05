# Roadmap

Goal: confirmed bugs in production LLM-serving engines, found by Phantom.

| Phase | Weeks | Deliverable | Done when |
|---|---|---|---|
| 0. Oracle + simulator | 1 | Toy engine, provenance oracle, seeded sim, replay | ✅ 0 false positives / 1k seeds; 2 plants caught |
| 1. Shrinking | 2 | Delta-debugging: failing seed → minimal event trace | ✅ ~50 events → median 6, max 13; 1-minimal (tested) |
| 2. Phantom in vLLM | 3–5 | Out-of-tree vLLM model on CPU backend writing tags through real block tables / slot mapping | Real vLLM passes clean seeds; a patched-in bug is caught |
| 3. Drive real vLLM deterministically | 6–8 | Seeded workload + abort/preempt/pressure injection against vLLM `EngineCore` | Same seed → same trace on real vLLM |
| 4. Disaggregation + faults | 9–11 | Two engines + simulated KV-transfer layer (delay, reorder, duplicate, abort mid-transfer, handshake failure) | ✅ v0.5.0: SGLang Oct-2026 slot race reproduced and caught; delay, swap, silent-drop, duplicate, abort mid-transfer, all under seeded fuzzing. Reorder, handshake failure and the NIXL class moved to Phase 5 (they need the NIXL connector) |
| 5. Bug hunt | 12–14 | Campaigns on vLLM main, SGLang, LMCache, llm-d | Issues filed with minimal repros; fix PRs |
| 6. Publish | 15–16 | Paper (arXiv/Zenodo), blog, meetup talk | — |

Highest-yield targets (newest, least-tested paths): push-mode KV transfer,
CPU offloading × quantized KV, hybrid Mamba/attention KV groups, LoRA + prefix cache.
