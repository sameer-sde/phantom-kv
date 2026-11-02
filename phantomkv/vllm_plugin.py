"""vLLM plugin entry point (group `vllm.general_plugins`, see pyproject.toml).

vLLM calls `register()` in every process it starts (API server, EngineCore,
workers), so the Phantom architecture is known everywhere. The model class is
given as a string so it is imported lazily, only when vLLM actually needs it.

If PHANTOM_VLLM_PLANT is set, the named bug is planted into vLLM in this
process as well (see vllm_plants.py). Unset, vLLM runs unmodified.
"""

import os


def register() -> None:
    from vllm import ModelRegistry

    if "PhantomForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "PhantomForCausalLM", "phantomkv.vllm_model:PhantomForCausalLM")

    plant = os.environ.get("PHANTOM_VLLM_PLANT")
    if plant:
        from .vllm_plants import apply

        apply(plant)
