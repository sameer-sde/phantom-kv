#!/usr/bin/env bash
# Real-vLLM acceptance checks. Run in CI (.github/workflows/vllm.yml) and locally:
#     bash scripts/ci_vllm.sh
# Every check must hold, in both directions:
#   * unmodified vLLM  -> Phantom reports nothing      (no false positives)
#   * planted vLLM bug -> Phantom reports a violation  (no false negatives)
# vllm_pd_check.py exits 0 when the outcome is the expected one (PASS without a
# fault, CAUGHT with one), so its checks are all `expect pass`.
set -uo pipefail

SEEDS="${PHANTOM_CI_SEEDS:-10}"
PLANT="stale-prefix-cache"
fail=0

expect() {  # expect <pass|fail> <description> <command...>
  local want="$1" what="$2"; shift 2
  echo "::group::${what}"
  "$@"
  local code=$?
  echo "::endgroup::"
  if { [ "$want" = pass ] && [ $code -eq 0 ]; } || { [ "$want" = fail ] && [ $code -eq 1 ]; }; then
    echo "OK   ${what} (exit ${code}, expected ${want})"
  else
    echo "FAIL ${what} (exit ${code}, expected ${want})"
    fail=1
  fi
}

expect pass "prefix-cache reuse, unmodified vLLM" \
  python scripts/vllm_phantom_check.py
expect pass "memory pressure, unmodified vLLM" \
  python scripts/vllm_phantom_check.py --pressure
expect fail "memory pressure, ${PLANT} planted -> must be caught" \
  python scripts/vllm_phantom_check.py --pressure --plant "${PLANT}"
expect pass "seeded fuzz x${SEEDS}, unmodified vLLM, each seed run twice" \
  python -m phantomkv vllm-fuzz --seeds "${SEEDS}" --check-determinism
expect fail "seeded fuzz x${SEEDS}, ${PLANT} planted -> must be caught" \
  python -m phantomkv vllm-fuzz --seeds "${SEEDS}" --plant "${PLANT}"

# Phase 4: two engines (prefill + decode) connected through PhantomFaultConnector
expect pass "two engines, no fault: transferred KV verified" \
  python scripts/vllm_pd_check.py
expect pass "two engines, swap fault -> must be caught" \
  python scripts/vllm_pd_check.py --fault swap
expect pass "two engines, silent-drop fault -> must be caught" \
  python scripts/vllm_pd_check.py --fault silent-drop
expect pass "two engines, abort while KV in flight, unmodified vLLM: must stay clean" \
  python scripts/vllm_pd_check.py --abort-mid-transfer
expect pass "two engines, abort while KV in flight, abort-free-race planted -> must be caught" \
  python scripts/vllm_pd_check.py --abort-mid-transfer --plant abort-free-race

# Phase 4, seeded: random load delays, abort timings and fillers on two engines
expect pass "PD fuzz x${SEEDS}, unmodified vLLM, each seed run twice" \
  python -m phantomkv pd-fuzz --seeds "${SEEDS}" --check-determinism
expect fail "PD fuzz x${SEEDS}, abort-free-race planted -> must be caught" \
  python -m phantomkv pd-fuzz --seeds "${SEEDS}" --plant abort-free-race
expect fail "PD fuzz x${SEEDS}, duplicate transfer fault -> must be caught" \
  python -m phantomkv pd-fuzz --seeds "${SEEDS}" --fault duplicate

# Phase 5: vLLM's CPU KV offloading (manager unmodified, copy engine replaced);
# stores take 40 steps and loads 1, on separate queues like vLLM's two streams
expect pass "CPU offload x${SEEDS}, unmodified vLLM, each seed run twice" \
  python -m phantomkv vllm-fuzz --profile cpu-offload-slow-stores --seeds "${SEEDS}" --check-determinism
expect fail "CPU offload x${SEEDS}, store-reported-early planted -> must be caught" \
  env PHANTOM_OFFLOAD_FAULT=store-reported-early \
  python -m phantomkv vllm-fuzz --profile cpu-offload-slow-stores --seeds "${SEEDS}"

exit $fail
