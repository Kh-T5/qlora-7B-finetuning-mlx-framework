"""Download Mistral-7B from Hugging Face and write it as a quantized checkpoint.

A one-time step. Decoder and MLP projections are quantized to 4 bits per row;
RMSNorm weights, the embedding and the LM head are stored full precision.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from mistral_qlora.checkpoint import (
    layer_norm_path,
    layer_weight_path,
    save_embeddings,
    save_norm,
    save_quantized_weight,
)
from mistral_qlora.config import Paths
from mistral_qlora.constants import (
    ATTN_PROJECTIONS,
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


def hf_weight_names(index: int) -> dict[str, str]:
    """Map each checkpoint name to its key in the Hugging Face state dict."""
    prefix = f"model.layers.{index}"
    names = {name: f"{prefix}.self_attn.{name}.weight" for name in ATTN_PROJECTIONS}
    names |= {name: f"{prefix}.mlp.{name}.weight" for name in MLP_PROJECTIONS}
    return names


def convert_layer(state_dict: dict, index: int, layers_dir: Path):
    """Write one decoder layer: quantized projections plus its two RMSNorms.

    Existing files are left alone so an interrupted run can resume.
    """
    for name, key in hf_weight_names(index).items():
        path = layer_weight_path(layers_dir, index, name)
        if path.exists():
            print(f"[skip] {path.name}")
            continue
        print(f"Quantizing {path.stem} {tuple(state_dict[key].shape)}")
        save_quantized_weight(
            path, *quantize_4bit_per_row(torch_to_mx_array(state_dict[key]))
        )

    for name in NORM_NAMES:
        save_norm(
            layer_norm_path(layers_dir, index, name),
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

    save_embeddings(
        paths.quantized_other,
        norm=torch_to_numpy(state_dict["model.norm.weight"]),
        embed=torch_to_numpy(state_dict["model.embed_tokens.weight"]),
        head=torch_to_numpy(state_dict["lm_head.weight"]),
    )

    print("Done.")


if __name__ == "__main__":
    main()
