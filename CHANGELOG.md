# Changelog

Each release marks a finished phase of the [roadmap](docs/ROADMAP.md). Day-by-day
details are in the [devlog](docs/DEVLOG.md).

## Unreleased (Phase 5: bug hunting)

- vLLM's `SimpleCPUOffloadConnector` on the CPU backend: its manager runs
  unmodified, only the CUDA copy engine is replaced. Six `vllm-fuzz`
  profiles (`cpu-offload*`); 0 violations so far, 3,753 blocks stored and
  317 loaded back and verified.
- Offload copy engine with separate load/store queues (like vLLM's two CUDA
  streams); planted `store-reported-early` caught; offloading in CI.
- `phantomkv vllm-diff`: run a seed twice, show the first step where the
  runs differ. Found that the harness stopped stepping while vLLM still had
  pending connector work; fixed.

## v0.5.0 (Phase 4: two-engine prefill/decode with transfer faults)

- Two real vLLM engines (prefill + decode) exchanging KV through vLLM's
  KV-connector API; Phantom verifies every transferred slot on the decode side.
- `PhantomFaultConnector`: planted transfer faults `swap` (KV of the wrong
  request) and `silent-drop` (load reported done, nothing written). Both caught.
- `PhantomAsyncFaultConnector`: asynchronous loads that land steps later,
  one-sided like RDMA.
- `abort-free-race` plant (the SGLang Oct-2026 bug class): an aborted request's
  blocks are freed while its KV is still in flight. Unmodified vLLM stays clean
  (4 late landings handled); with the plant, 54 `misplaced-block` violations and
  5 corrupted requests.
- Seeded PD fuzzing (`phantomkv pd-fuzz`, `pd-shrink`): each seed picks the
  load delay, transfers, abort timings and fillers. Unmodified vLLM: 0/20
  failing with 35 aborts while KV was in flight. `abort-free-race`: 14/20
  caught; a failing seed shrinks from 24 events to 2.
- Fourth transfer fault, `duplicate`: a stale copy of a KV write lands after
  the request's blocks may have been reused.
- `scripts/pd_campaign.sh`, 200 seeds per mode: unmodified vLLM 0/200;
  `abort-free-race` 115, `swap` 199, `silent-drop` 199, `duplicate` 184
  caught. The seed swap and silent-drop missed read no transferred KV at all.
- All two-engine runs are part of the `real-vllm` CI job.

## v0.4.0 (Phase 3: more chaos on real vLLM)

- vLLM configuration profiles: `baseline`, `chunked-prefill`, `preemption`,
  `block64`, with engine counters proving each stressed path was exercised.
- Second planted bug class, `missing-touch` (refcount). Found that vLLM's own
  assertions catch bugs that make its bookkeeping inconsistent; Phantom is
  needed for bugs that leave it consistent but wrong.
- In-process engine for exact step ordering and replay; honest verdicts
  (PASS / CAUGHT / VLLM-ASSERT / MISSED / CRASH) in `vllm_campaign.sh`.
- Engine errors recorded with file, line and source.

## v0.3.0 (Phase 2: Phantom inside real vLLM)

- `PhantomForCausalLM`: a weightless model registered through
  `vllm.general_plugins` that writes provenance tags into vLLM's real KV cache
  through its block tables and slot mappings, and verifies every read.
- Float32-exact tag encoding.
- Planted `stale-prefix-cache` bug in vLLM's block pool, caught on 15/20 seeds.
- Seeded step-by-step harness (`phantomkv vllm-fuzz`) and shrinking of
  real-vLLM failures (`phantomkv vllm-shrink`).
- `real-vllm` CI job running Phantom against a pinned vLLM CPU wheel.

## v0.2.0 (Phase 1: shrinking)

- Delta-debugging minimiser (ddmin): a failing seed becomes a minimal,
  1-minimal event list that still fails.
- `phantomkv shrink` command.

## v0.1.0 (Phase 0: oracle and deterministic simulator)

- Provenance tags and prefix chain hashes; an exact oracle with four
  violation kinds (`unwritten-read`, `misplaced-block`, `chain-break`,
  `slot-mismatch`).
- Toy paged-KV engine with prefix cache, preemption, PD transfer and aborts.
- Deterministic simulator driven by a single seed; `fuzz` and `replay` commands.
- CI running unit tests and a 1,000-seed clean fuzz on every push.
