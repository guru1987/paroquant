# Changes vs `z-lab/paroquant`

This fork applies two targeted patches to make `paroquant==0.1.13` usable on
consumer multi-GPU setups (e.g. 2× RTX 3090). Both patches are intended as
upstream PR candidates — once merged, this fork is no longer needed.

## 0.1.14 (this fork)

### 1. vLLM tensor-parallel support

**Bug:** `paroquant/inference/backends/vllm/plugin.py:_rotation_weight_loader`
does an unconditional `target.copy_(loaded_weight)`. For row-parallel layers
(`o_proj`, `down_proj`), vLLM allocates the rotation params with
`input_size_per_partition = input_size // tp_size`, but `loaded_weight` from
disk is the full input dim. The shape mismatch aborts model load:

```
RuntimeError: The size of tensor a (3072) must match the size of tensor b (6144)
at non-singleton dimension 1
```

(Numbers vary by model; example is `Qwen3.6-27B`'s `o_proj` at TP=2:
per-rank `6144 / 2 = 3072` vs full-size weight.)

**Fix:** add a `_maybe_shard_input` helper that detects the size mismatch and
slices `loaded_weight` along its last (input) dim by the current TP rank.
Column-parallel layers (where shapes already match) take a fast path with
no behavior change. TP=1 paths are unaffected (helper short-circuits).

**Verification:** on 2× RTX 3090 (sm_86) with `z-lab/Qwen3.6-27B-PARO` at
`--tensor-parallel-size 2`, model loads cleanly and inference matches TP=1
byte-for-byte on test prompts.

### 2. vLLM `vllm.general_plugins` auto-load

**Issue:** paroquant depends on two side-effect imports to take effect:

1. `paroquant.kernels.cuda` registers `torch.ops.rotation.rotate`.
2. `paroquant.inference.backends.vllm.plugin` registers
   `@register_quantization_config("paroquant")`.

In `0.1.13`, these imports are triggered only by launching
`python -m paroquant.cli.serve …` instead of vanilla `vllm serve …`. The shim
exists *solely* for those imports.

**Fix:** expose `paroquant.register()` as a `vllm.general_plugins` entry point
in `pyproject.toml`. vLLM auto-loads this group at startup, so vanilla
`vllm serve <model>` works without the wrapper.

```toml
[project.entry-points."vllm.general_plugins"]
paroquant = "paroquant:register"
```

```python
# paroquant/__init__.py
def register() -> None:
    """vLLM general-plugin entry point. Idempotent."""
    import paroquant.kernels.cuda
    import paroquant.inference.backends.vllm.plugin
```

The `paroquant.cli.serve` wrapper still works for back-compat.

## Hardware tested

- 2× NVIDIA RTX 3090 (sm_86, 24 GB), PCIe switch with P2P
- Driver 580.105.08, CUDA 13.0
- vLLM 0.19.1, PyTorch 2.10.0+cu130

## Unrelated note: `vllm/csrc/custom_all_reduce.cuh:455` on Ampere

vLLM's custom all-reduce kernel fails during CUDA-graph capture on consumer
Ampere cards (`Cuda error /workspace/csrc/custom_all_reduce.cuh:455 'invalid
argument'`). This is **not** caused by paroquant and is **not** fixed in this
fork. Workaround until vLLM fixes the kernel: launch with
`--disable-custom-all-reduce`.

---

**Authors:** [guru87](https://huggingface.co/guru87) ([GitHub: guru1987](https://github.com/guru1987)) and **Claude Opus 4.7** (Anthropic, 1M context). Diagnosis, patches, scripting, and docs were developed collaboratively over a single session in May 2026.
