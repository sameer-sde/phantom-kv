"""vLLM plugin entry point (group `vllm.general_plugins`, see pyproject.toml).

vLLM calls `register()` in every process it starts (API server, EngineCore,
workers), so the Phantom architecture is known everywhere. The model class is
given as a string so it is imported lazily, only when vLLM actually needs it.
"""


def register() -> None:
    from vllm import ModelRegistry

    if "PhantomForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "PhantomForCausalLM", "phantomkv.vllm_model:PhantomForCausalLM")
