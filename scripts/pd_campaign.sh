#!/usr/bin/env bash
# Phase 4 campaign: seeded two-engine (prefill/decode) fuzzing on real vLLM,
# unmodified and with every planted scheduler bug and transfer fault.
#     bash scripts/pd_campaign.sh              # 200 seeds per run
#     SEEDS=500 bash scripts/pd_campaign.sh
# Each run starts its own decode engine (~70 s on a laptop); the prefilled
# pool is reused. Prints one summary line per run; full logs go to
# campaign_logs/pd-*.log
set -uo pipefail

SEEDS="${SEEDS:-200}"
MODES="${MODES:-clean plant:abort-free-race fault:swap fault:silent-drop fault:duplicate}"
mkdir -p campaign_logs

printf "%-26s %-10s %-14s %s\n" DECODE-ENGINE RESULT FAILING-SEEDS EVIDENCE
for mode in $MODES; do
  log="campaign_logs/pd-${mode/:/-}.log"
  case "$mode" in
    clean)   args=(--check-determinism) ;;
    plant:*) args=(--plant "${mode#plant:}") ;;
    fault:*) args=(--fault "${mode#fault:}") ;;
  esac
  python -m phantomkv pd-fuzz --seeds "$SEEDS" "${args[@]}" >"$log" 2>&1
  code=$?
  failing=$(grep -oE "^[0-9]+ failing seeds" "$log" | grep -oE "^[0-9]+" || echo 0)
  evidence=$(grep -E "^evidence:" "$log" | sed 's/^evidence: //; s/async loads //')
  case "$mode:$code" in
    clean:0) verdict="PASS" ;;
    clean:*) verdict="ALARM" ;;        # problem on unmodified vLLM: investigate
    *:1)     verdict="CAUGHT" ;;       # Phantom reported a violation / wrong output
    *:2)     verdict="VLLM-ASSERT" ;;  # vLLM's own check fired; Phantom saw nothing
    *:0)     verdict="MISSED" ;;       # nothing detected at all
    *)       verdict="CRASH" ;;        # harness crashed: see the log
  esac
  grep -q "never fired\|was not exercised" "$log" && verdict="$verdict(!)"  # see the log
  label=$([ "$mode" = clean ] && echo unmodified || echo "$mode")
  printf "%-26s %-10s %-14s %s\n" "$label" "$verdict" "${failing}/${SEEDS}" "$evidence"
done
