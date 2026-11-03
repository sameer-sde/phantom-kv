#!/usr/bin/env bash
# Real-vLLM acceptance checks. Run in CI (.github/workflows/vllm.yml) and locally:
#     bash scripts/ci_vllm.sh
# Every check must hold, in both directions:
#   * unmodified vLLM  -> Phantom reports nothing      (no false positives)
#   * planted vLLM bug -> Phantom reports a violation  (no false negatives)
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

exit $fail
