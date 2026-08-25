#!/usr/bin/env python3
"""Load a double-quantized bitsandbytes GPT-2 checkpoint with mlx-nf4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx_nf4 as nf4
import numpy as np
from safetensors import safe_open


MODEL_ID = "manu02/gpt2-bnb-4bit-nf4-dq"
MODEL_REVISION = "7744ff22be99f562bdaa444612a35a20bf995999"
QUANTIZED_PROJECTION_COUNT = 48


def _set_module(root, path: str, value) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def _model_path(checkpoint_path: str) -> str:
    if not checkpoint_path.startswith("transformer."):
        raise ValueError(f"unexpected GPT-2 checkpoint path: {checkpoint_path}")
    return "model." + checkpoint_path[len("transformer.") :]


def _local_revision(model_dir: Path) -> str:
    metadata_path = (
        model_dir
        / ".cache"
        / "huggingface"
        / "download"
        / "model.safetensors.metadata"
    )
    if not metadata_path.is_file():
        raise RuntimeError(
            "the Hugging Face download revision receipt is missing; fetch the "
            "checkpoint with `hf download ... --revision ... --local-dir ...`"
        )
    return metadata_path.read_text().splitlines()[0].strip()


def load_gpt2(model_dir: Path):
    from mlx_lm.models.gpt2 import Model, ModelArgs

    config = json.loads((model_dir / "config.json").read_text())
    args = ModelArgs(
        model_type=config["model_type"],
        n_ctx=config["n_ctx"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        n_positions=config["n_positions"],
        layer_norm_epsilon=config["layer_norm_epsilon"],
        vocab_size=config["vocab_size"],
    )
    model = Model(args)
    checkpoint = model_dir / "model.safetensors"
    regular_weights = []
    quantized_modules = []

    with safe_open(checkpoint, framework="numpy") as sf:
        keys = set(sf.keys())
        quantized_bases = sorted(
            key.split(".quant_state.bitsandbytes__nf4")[0]
            for key in keys
            if key.endswith(".quant_state.bitsandbytes__nf4")
        )
        consumed = set()

        for base in quantized_bases:
            meta = json.loads(
                sf.get_tensor(f"{base}.quant_state.bitsandbytes__nf4")
                .tobytes()
                .decode("utf-8")
            )
            rows, cols = meta["shape"]
            group_size = int(meta.get("blocksize", 64))
            packed = sf.get_tensor(base).reshape(rows, cols // 2)

            observed_quant_map = mx.array(sf.get_tensor(f"{base}.quant_map"))
            codebook_error = mx.max(mx.abs(observed_quant_map - nf4.NF4_LUT)).item()
            if codebook_error > 1e-6:
                raise RuntimeError(
                    f"{base} uses an unexpected NF4 codebook (max diff {codebook_error})"
                )

            absmax = mx.array(sf.get_tensor(f"{base}.absmax"))
            nested_absmax_key = f"{base}.nested_absmax"
            nested_quant_map_key = f"{base}.nested_quant_map"
            if nested_absmax_key in keys and nested_quant_map_key in keys:
                scales = nf4.reconstruct_bitsandbytes_scales(
                    absmax,
                    nested_absmax=mx.array(sf.get_tensor(nested_absmax_key)),
                    nested_quant_map=mx.array(sf.get_tensor(nested_quant_map_key)),
                    nested_offset=float(meta["nested_offset"]),
                    nested_block_size=int(meta["nested_blocksize"]),
                )
            else:
                scales = nf4.reconstruct_bitsandbytes_scales(absmax)
            scales = mx.reshape(scales, (rows, cols // group_size))

            bias_key = base.replace(".weight", ".bias")
            bias = mx.array(sf.get_tensor(bias_key)) if bias_key in keys else None
            layer = nf4.NF4Linear.from_bitsandbytes(
                mx.array(packed), scales, bias=bias, group_size=group_size
            )
            module_path = _model_path(base.removesuffix(".weight"))
            _set_module(model, module_path, layer)
            quantized_modules.append((module_path, layer))

            consumed.add(base)
            for suffix in (
                ".absmax",
                ".quant_map",
                ".nested_absmax",
                ".nested_quant_map",
                ".nested_scale_offset",
                ".quant_state.bitsandbytes__nf4",
            ):
                consumed.add(base + suffix)
            consumed.add(bias_key)

        for key in sorted(keys - consumed):
            regular_weights.append((_model_path(key), mx.array(sf.get_tensor(key))))

    if len(quantized_modules) != QUANTIZED_PROJECTION_COUNT:
        raise RuntimeError(
            f"loaded {len(quantized_modules)} NF4 projections, expected "
            f"{QUANTIZED_PROJECTION_COUNT}"
        )
    model.load_weights(regular_weights, strict=False)
    mx.eval(model.parameters())
    return model, quantized_modules


def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, list[int]]:
    token_ids = tokenizer.encode(prompt).ids
    generated = list(token_ids)
    for _ in range(max_new_tokens):
        logits = model(mx.array([generated], dtype=mx.int32))
        next_token = int(mx.argmax(logits[0, -1]).item())
        generated.append(next_token)
        if next_token == 50256:
            break
    return tokenizer.decode(generated), generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    local_revision = _local_revision(args.model_dir)
    if local_revision != MODEL_REVISION:
        raise RuntimeError(
            f"checkpoint revision is {local_revision}, expected {MODEL_REVISION}"
        )

    load_start = time.perf_counter()
    model, quantized_modules = load_gpt2(args.model_dir)
    load_seconds = time.perf_counter() - load_start

    representative_name, representative = quantized_modules[0]
    parity_input = mx.reshape(mx.linspace(-0.5, 0.5, 5 * representative.input_dims), (1, 5, -1))
    native = representative(parity_input)
    reference = representative.reference_forward(parity_input)
    mx.eval(native, reference)
    parity_max_abs_diff = float(mx.max(mx.abs(native - reference)).item())

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    generation_start = time.perf_counter()
    text, token_ids = generate_greedy(
        model, tokenizer, args.prompt, args.max_new_tokens
    )
    generation_seconds = time.perf_counter() - generation_start

    model_file = args.model_dir / "model.safetensors"
    receipt = {
        "semantic_name": "Pinned GPT-2 bitsandbytes NF4 standalone-package smoke",
        "model_id": MODEL_ID,
        "model_revision": local_revision,
        "model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        "mlx_nf4_module": str(Path(nf4.__file__).resolve()),
        "quantized_projection_count": len(quantized_modules),
        "representative_projection": representative_name,
        "representative_parity_max_abs_diff": parity_max_abs_diff,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "token_ids": token_ids,
        "generated_text": text,
        "load_seconds": round(load_seconds, 6),
        "generation_seconds": round(generation_seconds, 6),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(text)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
