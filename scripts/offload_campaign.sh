#!/usr/bin/env bash
# Phase 5 target 2 campaign: vLLM's OffloadingConnector, every profile, plus
# the planted fault and the two storage failures vLLM must recover from.
#     bash scripts/offload_campaign.sh            # 50 seeds per run
#     SEEDS=200 bash scripts/offload_campaign.sh
# Clean runs replay every seed (--check-determinism). Logs: campaign_logs/off-*.log
set -uo pipefail

SEEDS="${SEEDS:-50}"
CLEAN="${CLEAN:-offload2 offload2-chunk2 offload2-chunk2-revisit offload2-chunk3-revisit offload2-slow-stores offload2-preempt offload2-small offload2-sync}"
# fault@profile : expect
FAULTS="${FAULTS:-store-reported-early@offload2-slow-stores:caught load-fails@offload2:recover store-fails@offload2:recover load-fails@offload2-chunk2-revisit:recover store-fails@offload2-chunk2-revisit:recover}"
mkdir -p campaign_logs

row() { printf "%-46s %-9s %-10s %s\n" "$1" "$2" "$3" "$4"; }
evidence() { grep -E "^cpu offload:" "$1" | sed 's/^cpu offload: //' \
  | grep -oE "(blocks_stored|blocks_loaded|partial_chunk_loads|partial_chunk_stores|sync_loads|loads_failed|stores_failed|flush_waits)=[0-9]+" | tr '\n' ' '; }
failing() { grep -oE "^[0-9]+ failing seeds" "$1" | grep -oE "^[0-9]+" || echo 0; }

row RUN RESULT FAILING EVIDENCE
for p in $CLEAN; do
  log="campaign_logs/off-$p.log"
  python -m phantomkv vllm-fuzz --profile "$p" --seeds "$SEEDS" --check-determinism >"$log" 2>&1
  case $? in 0) v=PASS ;; 1) v=ALARM ;; 2) v=VLLM-ASSERT ;; *) v=CRASH ;; esac
  row "unmodified $p" "$v" "$(failing "$log")/$SEEDS" "$(evidence "$log")"
done
for spec in $FAULTS; do
  fault=${spec%%@*}; rest=${spec#*@}; p=${rest%%:*}; expect=${rest#*:}
  log="campaign_logs/off-$p-$fault.log"
  PHANTOM_OFFLOAD2_FAULT=$fault python -m phantomkv vllm-fuzz --profile "$p" --seeds "$SEEDS" >"$log" 2>&1
  code=$?
  case "$expect:$code" in
    caught:1) v=CAUGHT ;; caught:0) v=MISSED ;;
    recover:0) v=RECOVERED ;; recover:1) v=FINDING ;;  # vLLM mishandled a storage failure
    *:2) v=VLLM-ASSERT ;; *) v=CRASH ;;
  esac
  row "$fault on $p" "$v" "$(failing "$log")/$SEEDS" "$(evidence "$log")"
done
