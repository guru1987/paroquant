# ParoQuant — community fork

**Pairwise Rotation Quantization for LLMs**, with multi-GPU and MTP fixes.

> [!NOTE]
> This is a community fork of [`z-lab/paroquant`](https://github.com/z-lab/paroquant).
> If you don't need the changes below, use upstream. Otherwise this fork installs
> alongside vanilla vLLM and works without any wrapper script.
>
> **Changes vs upstream** (full detail in [`docs/CHANGES.md`](docs/CHANGES.md)):
> - vLLM tensor-parallel support — `--tensor-parallel-size > 1` no longer crashes on row-parallel layers
> - vanilla `vllm serve <model>` works directly (auto-loaded as a `vllm.general_plugins` entry point — no `paroquant.cli.serve` shim needed)
> - new `paroquant-inject-mtp` CLI to wire missing MTP draft heads into paroquant checkpoints that ship without them
>
> Upstream PR pending: [z-lab/paroquant#41](https://github.com/z-lab/paroquant/pull/41).

## Install

```bash
pip install "vllm==0.19.1" "paroquant[vllm] @ git+https://github.com/guru1987/paroquant.git"
```

For CUDA 13.0 wheels, add the index URLs:
```bash
pip install "vllm==0.19.1" "paroquant[vllm] @ git+https://github.com/guru1987/paroquant.git" \
  --extra-index-url https://wheels.vllm.ai/0.19.1/cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

## Run a model

```bash
hf download z-lab/Qwen3.6-27B-PARO --local-dir Qwen3.6-27B-PARO
vllm serve ./Qwen3.6-27B-PARO --port 8000
```

For multi-GPU add `--tensor-parallel-size N`. On consumer Ampere (RTX 30xx/40xx), also add `--disable-custom-all-reduce` (vLLM cudagraph-capture bug, unrelated to paroquant). All other args pass through to vLLM.

## Speculative decoding (MTP) — fixing the missing draft head

Most paroquant Qwen3.5/3.6 checkpoints declare an MTP draft head in `config.json` but ship **without the weights**. Speculative decoding with these checkpoints starts the drafter from random init → 0% acceptance, pure overhead.

This fork ships `paroquant-inject-mtp` which transplants the official Qwen MTP head onto the paroquant base via a sharded-symlink layout (no LM weight duplication, ~tens-of-MiB-to-1-GiB extra on disk depending on size, fully reversible).

```bash
# Pick a paroquant model (left column) + matching MTP head (right column)
hf download z-lab/Qwen3.6-27B-PARO --local-dir Qwen3.6-27B-PARO
hf download guru87/Qwen3.6-27B-MTP --local-dir Qwen3.6-27B-MTP

paroquant-inject-mtp \
    --paro     ./Qwen3.6-27B-PARO \
    --mtp-from ./Qwen3.6-27B-MTP/mtp.safetensors \
    --output   ./Qwen3.6-27B-PARO-MTP

vllm serve ./Qwen3.6-27B-PARO-MTP \
    --tensor-parallel-size 2 --disable-custom-all-reduce \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

### MTP heads we publish

Each is a single-file extraction from the official BF16 base model, byte-identical (verified by SHA256) to MTP weights shipped in other community quants. SHA256SUMS included for audit.

| paroquant model                                                                                      | MTP head repo                                                                                  | size    |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|---------|
| [`z-lab/Qwen3.5-0.8B-PARO`](https://huggingface.co/z-lab/Qwen3.5-0.8B-PARO)                          | [`guru87/Qwen3.5-0.8B-MTP`](https://huggingface.co/guru87/Qwen3.5-0.8B-MTP)                     | 39 MiB  |
| [`z-lab/Qwen3.5-2B-PARO`](https://huggingface.co/z-lab/Qwen3.5-2B-PARO)                              | [`guru87/Qwen3.5-2B-MTP`](https://huggingface.co/guru87/Qwen3.5-2B-MTP)                         | 116 MiB |
| [`z-lab/Qwen3.5-4B-PARO`](https://huggingface.co/z-lab/Qwen3.5-4B-PARO)                              | [`guru87/Qwen3.5-4B-MTP`](https://huggingface.co/guru87/Qwen3.5-4B-MTP)                         | 230 MiB |
| [`z-lab/Qwen3.5-9B-PARO`](https://huggingface.co/z-lab/Qwen3.5-9B-PARO)                              | [`guru87/Qwen3.5-9B-MTP`](https://huggingface.co/guru87/Qwen3.5-9B-MTP)                         | 464 MiB |
| [`z-lab/Qwen3.6-27B-PARO`](https://huggingface.co/z-lab/Qwen3.6-27B-PARO)                            | [`guru87/Qwen3.6-27B-MTP`](https://huggingface.co/guru87/Qwen3.6-27B-MTP)                       | 811 MiB |

> [!NOTE]
> `z-lab/Qwen3.5-35B-A3B-PARO` (MoE) **doesn't work cleanly under TP > 1 in current paroquant** — it's not the MTP, the base model itself can't load: paroquant's plugin only registers a `LinearMethod`, not the `MoEMethodBase` that vLLM's `FusedMoE` expects for stacked-experts checkpoints. Tracked as a known limitation; needs upstream work in paroquant.

## What does each model size give you?

Measured single-stream greedy on RTX 3090 with this fork + injected MTP, `num_speculative_tokens=2`:

| size  | tok/s | MTP accept | KV cache @ 8k context | minimum GPU |
|-------|------:|-----------:|----------------------:|-------------|
| 0.8B  |   212 |       54 % |              326 k tok | 1× 24 GB     |
| 2B    |   191 |       55 % |                       | 1× 24 GB     |
| 4B    |   171 |       75 % |                       | 1× 24 GB     |
| 9B    |   120 |       72 % |                       | 1× 24 GB     |
| 27B   |    75 |       73 % |                       | **2× 24 GB** |

Concurrent load tested on the 27B at parallel=24, 1k-in / 2k-out → **580 tok/s sustained** with 82 % KV utilization (vs ~450 tok/s on Qwen3.6-27B-GPTQ-8bit at the same hardware/workload).

## Citation (upstream paper)

```bibtex
@inproceedings{liang2026paroquant,
  title     = {{ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference}},
  author    = {Liang, Yesheng and Chen, Haisheng and Zhang, Zihan and Han, Song and Liu, Zhijian},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

## License

MIT, inherited from upstream paroquant. Models hosted on HuggingFace inherit Apache-2.0 from their respective Qwen base models.

## Quantize your own model

Out of scope for this fork — see [upstream's instructions](https://github.com/z-lab/paroquant#quantize-your-own-model). The fork only patches inference; the optimization/conversion path is unchanged.

---

**Authors:** [guru87](https://huggingface.co/guru87) ([GitHub: guru1987](https://github.com/guru1987)) and **Claude Opus 4.7** (Anthropic, 1M context). Diagnosis, patches, scripting, and docs were developed collaboratively over a single session in May 2026.
