"""PhantomForCausalLM: a weightless model that runs inside real vLLM.

Imported only by vLLM (via the plugin in `vllm_plugin.py`), never by the rest
of phantomkv, so the core package stays dependency-free.

Day 4 scope: vLLM must accept this class as a text-generation model, allocate
a real paged KV cache for its single attention layer, and run forward passes.
The forward pass does not touch the KV cache yet; provenance tags arrive on Day 5.
See docs/VLLM_INTEGRATION.md section 3 for the full design.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention

# The CPU attention backend's smallest supported head size
# (vllm/v1/attention/backends/cpu_attn.py: get_supported_head_sizes).
HEAD_SIZE = 32
# Day 4 placeholder: every step emits this token. Day 5 replaces it with
# a hash of the provenance chain read back from the KV cache.
PLACEHOLDER_TOKEN = 1000


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
        n = positions.shape[0]
        return torch.zeros(n, self.hidden_size, dtype=torch.float32, device=positions.device)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.full((hidden_states.shape[0], self.vocab_size), -1e4,
                            dtype=torch.float32, device=hidden_states.device)
        logits[:, PLACEHOLDER_TOKEN] = 1e4
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return set()  # weightless by design
