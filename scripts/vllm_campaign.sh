#!/usr/bin/env bash
# Phase 3 campaign: every vLLM profile, unmodified and with a planted bug.
#     bash scripts/vllm_campaign.sh            # 20 seeds per run
#     SEEDS=50 bash scripts/vllm_campaign.sh
# Each run starts its own vLLM engine (~80 s on a laptop), so 4 profiles x 2
# takes ~12-15 minutes. Prints one summary line per run; full logs go to
# campaign_logs/.
set -uo pipefail

SEEDS="${SEEDS:-20}"
PLANT="stale-prefix-cache"
PROFILES="${PROFILES:-baseline chunked-prefill preemption block64}"
mkdir -p campaign_logs

printf "%-16s %-20s %-10s %s\n" PROFILE VLLM RESULT DETAILS
for profile in $PROFILES; do
  for mode in clean planted; do
    log="campaign_logs/${profile}-${mode}.log"
    if [ "$mode" = clean ]; then
      python -m phantomkv vllm-fuzz --profile "$profile" --seeds "$SEEDS" --check-determinism >"$log" 2>&1
    else
      python -m phantomkv vllm-fuzz --profile "$profile" --seeds "$SEEDS" --plant "$PLANT" >"$log" 2>&1
    fi
    code=$?
    summary=$(grep -E "^phantomkv vllm:" "$log" | sed 's/^phantomkv vllm: //')
    counters=$(grep -E "^engine counters:" "$log" | sed 's/^engine counters: //')
    failing=$(grep -E "failing seeds" "$log" | sed 's/;.*//')
    case "$mode:$code" in
      clean:0)   verdict="PASS" ;;
      planted:1) verdict="CAUGHT" ;;
      clean:*)   verdict="ALARM" ;;    # violation on unmodified vLLM: investigate
      planted:*) verdict="MISSED" ;;   # planted bug not caught
    esac
    label=$([ "$mode" = clean ] && echo unmodified || echo "$PLANT")
    printf "%-16s %-20s %-10s %s | %s %s\n" "$profile" "$label" "$verdict" "$summary" "$counters" "$failing"
  done
done
