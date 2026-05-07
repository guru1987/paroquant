#!/usr/bin/env python3
"""
inject_mtp.py — wire an external MTP head into a paroquant-quantized Qwen3.6 model.

Background
----------
`z-lab/Qwen3.6-27B-PARO` (and likely siblings) ship without the multi-token-prediction
draft head despite `mtp_num_hidden_layers=1` declared in `config.json`. As a result,
launching vLLM with `--speculative-config '{"method": "mtp", ...}'` initializes the
draft module from random weights and the verifier rejects every drafted token (0%
acceptance, no speedup, just overhead).

The MTP head from any pre-existing Qwen3.6-27B variant whose authors kept it in
BF16 (we use the public `Qwen/Qwen3.6-27B-FP8` `mtp.safetensors` shipped alongside
the FP8 quant, OR a community quant like `Qwen3.6-27B-GPTQ-8bit` that kept MTP in
BF16) transplants cleanly onto the paroquant base — the MTP layer's prediction
head only needs the base model's hidden states, and paroquant's INT4 hidden states
are close enough to BF16 for ~85-92% draft acceptance in practice.

Output
------
A new model directory is produced as a *sharded* safetensors layout:
  - `model-00001-of-00002.safetensors`  — symlink to the original paroquant weights
  - `model-00002-of-00002.safetensors`  — newly written, MTP tensors only
  - `model.safetensors.index.json`      — updated weight_map covering both shards
  - All other files (config, tokenizer, ...) are symlinked from the source.

This means injection is fast (~seconds), uses ~1 GB of disk, and is fully
reversible (delete the output directory).

Usage
-----
    inject_mtp.py \\
        --paro /path/to/Qwen3.6-27B-PARO \\
        --mtp-from /path/to/Qwen3.6-27B-FP8           \\
        --output /path/to/Qwen3.6-27B-PARO-MTP

    # Or supply a single .safetensors file containing only mtp.* keys:
    inject_mtp.py --paro ... --mtp-from /path/to/mtp.safetensors --output ...

Verification
------------
    inject_mtp.py --verify /path/to/Qwen3.6-27B-PARO-MTP
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

def _require_deps():
    """Import torch + safetensors lazily so `--help` works without them installed."""
    try:
        import torch  # noqa: F401
        from safetensors import safe_open  # noqa: F401
        from safetensors.torch import save_file  # noqa: F401
    except ImportError as e:
        sys.stderr.write(f"missing dep: {e}; install with: pip install torch safetensors\n")
        sys.exit(2)


REQUIRED_MTP_KEYS = {
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
}


def _safetensors_files(path: Path) -> list:
    """Return all .safetensors files under `path`, or `[path]` if `path` is itself one."""
    if path.is_file() and path.suffix == ".safetensors":
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix == ".safetensors")
    raise ValueError(f"not a safetensors file or dir: {path}")


def collect_mtp(source: Path) -> dict:
    """Read every `mtp.*` tensor (eagerly, to memory) from a safetensors file or sharded dir.

    If a quantized source is given (FP8 / AWQ / etc.) we accept any tensor whose
    name starts with `mtp.`, including weight_packed/scale_inv variants — but
    those are only useful if the consumer architecture matches. For paroquant
    target use BF16-keep-MTP sources (e.g. Qwen3.6-27B-GPTQ-8bit, or the
    `mtp.safetensors` from Qwen/Qwen3.6-27B-FP8 if you accept FP8 drafter weights).
    """
    from safetensors import safe_open  # local import after _require_deps()
    out: dict = {}
    for f in _safetensors_files(source):
        with safe_open(f, framework="pt") as st:
            for k in st.keys():
                if k.startswith("mtp."):
                    out[k] = st.get_tensor(k)
    if not out:
        raise RuntimeError(f"no mtp.* tensors found in {source}")
    return out


def list_paroquant_keys(paro_safetensors: Path) -> list:
    """Enumerate all tensor keys in the paroquant single-file checkpoint."""
    from safetensors import safe_open
    if not paro_safetensors.is_file():
        raise FileNotFoundError(paro_safetensors)
    with safe_open(paro_safetensors, framework="pt") as st:
        return list(st.keys())


def build_dir(paro: Path, mtp_src: Path, output: Path, *, force: bool = False) -> None:
    from safetensors.torch import save_file
    paro_st = paro / "model.safetensors"
    if not paro_st.is_file():
        raise FileNotFoundError(
            f"expected single-file paroquant checkpoint at {paro_st} "
            f"(this script targets the paroquant export layout)"
        )

    if output.exists() and not force:
        if any(output.iterdir()):
            raise RuntimeError(
                f"{output} exists and is non-empty; pass --force to overwrite or pick a different path"
            )
    output.mkdir(parents=True, exist_ok=True)

    # 1. Symlink everything from paroquant dir, except the safetensors and any old index.
    for entry in paro.iterdir():
        dst = output / entry.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if entry.name == "model.safetensors":
            os.symlink(entry, output / "model-00001-of-00002.safetensors")
        elif entry.name.startswith("model.safetensors.index"):
            continue  # we will write a fresh index
        else:
            os.symlink(entry, dst)

    # 2. Collect MTP tensors from the source.
    print(f"[inject_mtp] reading MTP tensors from {mtp_src}")
    mtp_tensors = collect_mtp(mtp_src)
    missing = REQUIRED_MTP_KEYS - set(mtp_tensors)
    if missing:
        print(
            f"[inject_mtp] WARN: source is missing {len(missing)} of the standard MTP keys; "
            f"draft acceptance may be impaired:\n  {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}",
            file=sys.stderr,
        )
    total_bytes = sum(t.numel() * t.element_size() for t in mtp_tensors.values())
    print(f"[inject_mtp] MTP tensors: {len(mtp_tensors)}  ({total_bytes / 2**30:.2f} GiB)")

    # 3. Write MTP-only shard.
    out_shard = output / "model-00002-of-00002.safetensors"
    save_file(mtp_tensors, str(out_shard))
    print(f"[inject_mtp] wrote {out_shard}")

    # 4. Write index.json.
    paro_keys = list_paroquant_keys(paro_st)
    weight_map: dict[str, str] = {}
    weight_map.update({k: "model-00001-of-00002.safetensors" for k in paro_keys})
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in mtp_tensors})

    index = {
        "metadata": {
            "total_size": paro_st.stat().st_size + total_bytes,
            "produced_by": "inject_mtp.py (paroquant-fixes)",
        },
        "weight_map": weight_map,
    }
    with open(output / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(
        f"[inject_mtp] wrote model.safetensors.index.json "
        f"({len(weight_map)} keys: {len(paro_keys)} paroquant + {len(mtp_tensors)} MTP)"
    )

    print(f"[inject_mtp] done -> {output}")
    print(
        "[inject_mtp] launch vLLM with:\n"
        f"  --model {output} \\\n"
        f"  --speculative-config '{{\"method\": \"mtp\", \"num_speculative_tokens\": 2}}'"
    )


def verify(output: Path) -> int:
    """Sanity-check a previously injected directory. Returns 0 if OK, nonzero on issues."""
    from safetensors import safe_open
    rc = 0
    idx_path = output / "model.safetensors.index.json"
    if not idx_path.is_file():
        print(f"FAIL: no index at {idx_path}")
        return 2
    idx = json.load(open(idx_path))
    weight_map: dict[str, str] = idx["weight_map"]
    print(f"index keys: {len(weight_map)}")

    # Confirm each declared shard exists.
    shards = set(weight_map.values())
    for s in shards:
        p = output / s
        if not p.exists():
            print(f"FAIL: shard {s} declared in index but not present")
            rc = 2

    # Confirm every required MTP key is in the index.
    for k in sorted(REQUIRED_MTP_KEYS):
        if k not in weight_map:
            print(f"FAIL: required MTP key missing from index: {k}")
            rc = 2

    # Open each shard and confirm the keys it claims to hold are actually there.
    by_shard: dict = {}
    for k, s in weight_map.items():
        by_shard.setdefault(s, set()).add(k)
    for s, declared in by_shard.items():
        with safe_open(output / s, framework="pt") as st:
            actual = set(st.keys())
        missing = declared - actual
        extra = actual - declared
        if missing:
            print(f"FAIL: {s} index says {len(missing)} keys present that aren't (sample: {next(iter(missing))})")
            rc = 2
        if extra:
            print(f"WARN: {s} contains {len(extra)} keys not listed in index (harmless)")

    if rc == 0:
        print("OK: index ↔ shard contents consistent; MTP keys present")
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inject_mtp.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--paro", type=Path, help="paroquant model dir (with model.safetensors)")
    p.add_argument(
        "--mtp-from",
        type=Path,
        help="source containing mtp.* tensors: dir or single .safetensors file",
    )
    p.add_argument("--output", type=Path, help="destination dir for the MTP-augmented model")
    p.add_argument("--force", action="store_true", help="overwrite non-empty --output")
    p.add_argument("--verify", type=Path, help="verify a previously injected dir, then exit")

    args = p.parse_args(argv)

    _require_deps()

    if args.verify is not None:
        return verify(args.verify)

    for required in ("paro", "mtp_from", "output"):
        if getattr(args, required) is None:
            p.error(f"--{required.replace('_', '-')} is required (or pass --verify <dir>)")

    build_dir(args.paro, args.mtp_from, args.output, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
