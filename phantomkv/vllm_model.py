"""PhantomForCausalLM: a weightless model that runs inside real vLLM.

Imported only by vLLM (via the plugin in `vllm_plugin.py`), never by the rest
of phantomkv, so the core package stays dependency-free.

Every forward step:
  1. WRITE (all requests first, ascending position): for each new token at
     position p, compute
     chain[p] = H(chain[p-1], token) -- chain[p-1] is READ BACK from the KV
     slot holding position p-1 -- and write the tag into slot_mapping[t].
  2. CHECK (only after every write): walk each request's block table over its whole context and verify
     every slot: tag present, right position, hash chain unbroken. Also check
     that the slot vLLM told us to write equals the slot its block table implies.
  3. EMIT: hidden_states[t, 0] = next token for that position; compute_logits
     turns it into a one-hot, so greedy sampling returns it exactly.

Violations are appended as JSON lines to $PHANTOM_REPORT (default
phantom_report.jsonl), because the model may run in a different process than
the harness. See docs/VLLM_INTEGRATION.md section 3.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import get_attention_context

from . import tags
from .provenance import ROOT, chain_step

# Smallest head size the CPU attention backend supports (cpu_attn.py).
HEAD_SIZE = 32


class PhantomForCausalLM(nn.Module):
    """One real Attention layer (so vLLM allocates paged KV for it), no weights."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        self.vocab_size: int = cfg.vocab_size
        self.hidden_size: int = cfg.hidden_size
        self.layer_name = f"{prefix + '.' if prefix else ''}model.layers.0.attn"
        # Never called as a kernel: it exists so the engine builds a KV cache
        # spec, allocates the paged tensor, binds it and builds metadata for it.
        self.attn = Attention(
            num_heads=1,
            head_size=HEAD_SIZE,
            scale=1.0,
            num_kv_heads=1,
            cache_config=vllm_config.cache_config,
            prefix=self.layer_name,
        )
        self.report_path = os.environ.get("PHANTOM_REPORT", "phantom_report.jsonl")
        self.stats = {"steps": 0, "tokens_written": 0, "slots_checked": 0, "violations": 0,
                      # evidence counters (not checks): did the engine do what we think?
                      "prefill_continuations": 0,  # multi-token step resuming a previous one
                      "max_step_tokens": 0}
        # (first block id, next position) of every multi-token write last step;
        # used to recognise a prompt being prefilled in chunks across steps.
        self._prev_chunk_ends: set[tuple[int, int]] = set()

    # ------------------------------------------------------------ vLLM API
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(input_ids.shape[0], self.hidden_size, dtype=torch.float32,
                           device=input_ids.device)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden = torch.zeros(positions.shape[0], self.hidden_size, dtype=torch.float32,
                             device=positions.device)
        meta, _, kv_cache, slot_mapping = get_attention_context(self.layer_name)
        if meta is None or kv_cache.numel() == 0 or input_ids is None:
            return hidden  # warm-up / profiling run: no real requests
        if slot_mapping is None:
            slot_mapping = meta.slot_mapping
        self._step(meta, kv_cache, slot_mapping, input_ids, positions, hidden)
        return hidden

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.full((hidden_states.shape[0], self.vocab_size), -1e4,
                            dtype=torch.float32, device=hidden_states.device)
        tok = hidden_states[:, 0].round().long().clamp(0, self.vocab_size - 1)
        logits[torch.arange(hidden_states.shape[0]), tok] = 1e4
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return set()  # weightless by design

    # ------------------------------------------------------------- Phantom
    def _step(self, meta, kv_cache, slot_mapping, input_ids, positions, hidden) -> None:
        # CPU KV layout: [num_blocks, num_kv_heads, block_size, 2 * head_size]
        assert kv_cache.dim() == 4 and kv_cache.shape[1] == 1, tuple(kv_cache.shape)
        B = kv_cache.shape[2]
        n_tok = int(meta.num_actual_tokens)
        qsl = meta.query_start_loc.tolist()
        seq_lens = meta.seq_lens.tolist()
        bt = meta.block_table
        ids = input_ids[:n_tok].tolist()
        pos = positions[:n_tok].tolist()
        slots = slot_mapping[:n_tok].tolist()
        self.stats["steps"] += 1

        def read(slot: int):
            return tags.decode(kv_cache[slot // B, 0, slot % B, :tags.TAG_WIDTH].tolist())

        def write(slot: int, vals: list[float]) -> None:
            kv_cache[slot // B, 0, slot % B, :tags.TAG_WIDTH] = torch.tensor(
                vals, dtype=kv_cache.dtype)

        rows = [bt[r].tolist() for r in range(len(seq_lens))]

        def slot_of(r: int, p: int) -> int:
            return rows[r][p // B] * B + p % B

        # 1) WRITE every new token in the batch before ANY read, like a real
        #    attention layer (all KV writes for the batch, then attention).
        #    vLLM caches blocks at schedule time, so request Y may hit blocks
        #    that request X fills in this very step; Y's row can come first in
        #    the batch. Writing in ascending position order guarantees the slot
        #    for p-1 is final before the token at p reads it, across requests.
        new_tokens = []  # (pos, batch_row, flat_index)
        for r in range(len(seq_lens)):
            for t in range(qsl[r], qsl[r + 1]):
                new_tokens.append((pos[t], r, t))
        for p, r, t in sorted(new_tokens):
            tok, s = ids[t], slots[t]
            if s != slot_of(r, p):
                self._violate("slot-mismatch", r, p,
                              f"slot_mapping={s} but block table implies {slot_of(r, p)}")
            prev = ROOT
            if p > 0:
                tag = read(slot_of(r, p - 1))
                prev = tag[2] if tag else ROOT
            chain = chain_step(prev, tok)
            write(s, tags.encode(p, tok, chain))
            hidden[t, 0] = float(tags.next_token(chain, self.vocab_size))
            self.stats["tokens_written"] += 1

        # Evidence of chunked prefill: a request writing several tokens this step
        # starting exactly where its previous multi-token write ended.
        chunk_ends: set[tuple[int, int]] = set()
        for r in range(len(seq_lens)):
            q_len = qsl[r + 1] - qsl[r]
            if q_len > 1 and rows[r]:
                start = pos[qsl[r]]
                if (rows[r][0], start) in self._prev_chunk_ends:
                    self.stats["prefill_continuations"] += 1
                chunk_ends.add((rows[r][0], start + q_len))
        self._prev_chunk_ends = chunk_ends
        self.stats["max_step_tokens"] = max(self.stats["max_step_tokens"], n_tok)

        # 2) CHECK each request's whole context through the block table
        for r in range(len(seq_lens)):
            prev = ROOT
            for q in range(int(seq_lens[r])):
                tag = read(slot_of(r, q))
                self.stats["slots_checked"] += 1
                if tag is None:
                    self._violate("unwritten-read", r, q, f"slot {slot_of(r, q)} holds no tag")
                    break
                tpos, ttok, tchain = tag
                if tpos != q:
                    self._violate("misplaced-block", r, q,
                                  f"slot {slot_of(r, q)} holds position {tpos}")
                    break
                if tchain != chain_step(prev, ttok):
                    self._violate("chain-break", r, q,
                                  f"slot {slot_of(r, q)} does not extend the prefix before it")
                    break
                prev = tchain
        self._flush_stats()

    def _violate(self, kind: str, req_row: int, pos: int, detail: str) -> None:
        self.stats["violations"] += 1
        rec = {"kind": kind, "step": self.stats["steps"], "batch_row": req_row,
               "pos": pos, "detail": detail}
        with open(self.report_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _flush_stats(self) -> None:
        with open(self.report_path + ".stats.json", "w") as f:
            json.dump(self.stats, f)
