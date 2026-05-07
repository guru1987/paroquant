"""ParoQuant — Pairwise Rotation Quantization for LLMs.

The :func:`register` callable below is exposed as a ``vllm.general_plugins``
entry point in ``pyproject.toml`` so that vanilla ``vllm serve`` (or any other
vLLM entrypoint) auto-loads paroquant's quantization backend at startup.
End-users no longer need the ``paroquant.cli.serve`` shim solely for the
side-effect of importing the rotation kernel and quant config registration.

Importing the modules below triggers their decorators
(``@register_quantization_config("paroquant")``) and registers the
``torch.ops.rotation.rotate`` custom op.
"""


def register() -> None:
    """vLLM general-plugin entry point. Idempotent."""
    # Import for side effects only — the heavy lifting happens via
    # decorators inside these modules.
    import paroquant.kernels.cuda  # noqa: F401 — registers torch.ops.rotation.rotate
    import paroquant.inference.backends.vllm.plugin  # noqa: F401 — registers ParoQuantConfig
