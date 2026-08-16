"""Download Mistral-7B from Hugging Face and write it as a quantized checkpoint.

A one-time step. Decoder and MLP projections are quantized to 4 bits per row;
RMSNorm weights, the embedding and the LM head are stored full precision.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from mistral_qlora.config import Paths
from mistral_qlora.constants import (
    ATTN_PROJECTIONS,
    LAYER_NORM_TEMPLATE,
    LAYER_WEIGHT_TEMPLATE,
    MLP_PROJECTIONS,
    MODEL_NAME,
    NORM_NAMES,
)
from mistral_qlora.quant.quant_4bit import quantize_4bit_per_row


def torch_to_mx_array(tensor: torch.Tensor) -> mx.array:
    """Convert a torch tensor to an MLX array."""
    return mx.array(tensor.detach().to("cpu").numpy())


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy array."""
    return tensor.detach().to("cpu").numpy()


def save_quantized(weight: torch.Tensor, path: Path):
    """Quantize one linear weight to 4 bits per row and write it as .npz.

    Biases are not quantized; the projections converted here carry none.
    Existing files are left alone so an interrupted run can resume.
    """
    if path.exists():
        print(f"[skip] {path.name} already exists.")
        return

    print(f"Quantizing {path.stem} with shape {tuple(weight.shape)}")
    weight_q, scale, row_min, orig_cols = quantize_4bit_per_row(
        torch_to_mx_array(weight)
    )

    np.savez(
        path,
        weight_q=np.array(weight_q, copy=False, dtype=np.uint8),
        scale=np.array(scale, copy=False),
        row_min=np.array(row_min, copy=False),
        orig_in=np.int32(orig_cols),
    )


def save_unquantized(
    norm: torch.Tensor, embed: torch.Tensor, head: torch.Tensor, path: Path
):
    """Write the final norm, embedding and LM head to a single .npz."""
    if path.exists():
        print(f"[skip] {path.name} already exists.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        norm_np=torch_to_numpy(norm),
        embed_np=torch_to_numpy(embed),
        head_np=torch_to_numpy(head),
    )


def hf_weight_names(index: int) -> dict[str, str]:
    """Map each checkpoint name to its key in the Hugging Face state dict."""
    prefix = f"model.layers.{index}"
    names = {name: f"{prefix}.self_attn.{name}.weight" for name in ATTN_PROJECTIONS}
    names |= {name: f"{prefix}.mlp.{name}.weight" for name in MLP_PROJECTIONS}
    return names


def convert_layer(state_dict: dict, index: int, layers_dir: Path):
    """Write one decoder layer: quantized projections plus its two RMSNorms."""
    for name, key in hf_weight_names(index).items():
        save_quantized(
            state_dict[key],
            layers_dir / LAYER_WEIGHT_TEMPLATE.format(index=index, name=name),
        )

    for name in NORM_NAMES:
        path = layers_dir / LAYER_NORM_TEMPLATE.format(index=index, name=name)
        np.save(
            path,
            torch_to_numpy(state_dict[f"model.layers.{index}.{name}_layernorm.weight"]),
        )


def main():
    paths = Paths()
    layers_dir = paths.quantized_layers
    layers_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HF model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    state_dict = model.state_dict()

    for index in range(model.config.num_hidden_layers):
        convert_layer(state_dict, index, layers_dir)

    save_unquantized(
        norm=state_dict["model.norm.weight"],
        embed=state_dict["model.embed_tokens.weight"],
        head=state_dict["lm_head.weight"],
        path=paths.quantized_other,
    )

    print("Done.")


if __name__ == "__main__":
    main()
