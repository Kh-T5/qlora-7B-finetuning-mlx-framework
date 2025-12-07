import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM
import mlx.core as mx
from src.quant.quant_4bit import quantize_4bit_per_row
from src.config import (
    MODEL_NAME,
    mistral_other_layers_quant_path,
    mistral_decoder_layers_quant_dir,
)


def torch_to_mx_array(tensor: torch.Tensor) -> mx.array:
    """
    Convert a torch tensor to an MLX array
    """
    tensor = tensor.detach().to("cpu")
    np_arr = tensor.numpy()
    return mx.array(np_arr)


def save_linear(
    norm_t: torch.Tensor,
    embed_t: torch.Tensor,
    head_t: torch.Tensor,
    filepath: str,
):
    if os.path.exists(filepath):
        print(f"[skip] {filepath} already exists.")
        return

    norm_t = norm_t.detach().to("cpu")
    norm_np = norm_t.numpy()

    embed_t = embed_t.detach().to("cpu")
    embed_np = embed_t.numpy()

    head_t = head_t.detach().to("cpu")
    head_np = head_t.numpy()

    np.savez(filepath, norm_np=norm_np, embed_np=embed_np, head_np=head_np)


def save_linear_quantized(
    name: str,
    weight_t: torch.Tensor,
    out_dir: str,
):
    """
    Quantize a linear weight and save it (plus optional bias) to an .npz file.

    name:     Quantized layer's name
    weight_t: torch.Tensor (out_features x in_features)
    """
    file_path = os.path.join(out_dir, f"{name}.npz")
    if os.path.exists(file_path):
        print(f"[skip] {file_path} already exists.")
        return

    print(f"Quantizing {name} with shape {tuple(weight_t.shape)}")

    ### Torch tensor -> mx.array -> quantization 4-bit
    W_mx = torch_to_mx_array(weight_t)
    weight_q, scale, row_min, orig_cols = quantize_4bit_per_row(W_mx)

    ### Weights saved as .npz --> need numpy array
    weight_q_np = np.array(weight_q, copy=False)
    scale_np = np.array(scale, copy=False)
    row_min_np = np.array(row_min, copy=False)
    orig_in_np = np.int32(orig_cols)

    ### We do not quantize bias since it has small size

    np.savez(
        file_path,
        weight_q=weight_q_np,
        scale=scale_np,
        row_min=row_min_np,
        orig_in=orig_in_np,
    )


def main():
    if not os.path.exists(mistral_decoder_layers_quant_dir):
        os.makedirs(mistral_decoder_layers_quant_dir)

    print(f"Loading HF model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    sd = model.state_dict()

    num_layers = model.config.num_hidden_layers

    ### Quantization of DECODER and MLP layers before saving
    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        ### ------------- Decoder -----------------
        # Querie proj
        w_name = f"{prefix}.self_attn.q_proj.weight"
        weight_t = sd[w_name]

        layer_tag = f"layer_{i:02d}_q_proj"
        save_linear_quantized(layer_tag, weight_t, mistral_decoder_layers_quant_dir)

        # Values proj
        w_name = f"{prefix}.self_attn.v_proj.weight"
        weight_t = sd[w_name]

        layer_tag = f"layer_{i:02d}_v_proj"
        save_linear_quantized(layer_tag, weight_t, mistral_decoder_layers_quant_dir)

        # Keys proj
        w_name = f"{prefix}.self_attn.k_proj.weight"
        weight_t = sd[w_name]

        layer_tag = f"layer_{i:02d}_k_proj"
        save_linear_quantized(layer_tag, weight_t, mistral_decoder_layers_quant_dir)

        # O proj
        w_name = f"{prefix}.self_attn.o_proj.weight"
        weight_t = sd[w_name]

        layer_tag = f"layer_{i:02d}_o_proj"
        save_linear_quantized(layer_tag, weight_t, mistral_decoder_layers_quant_dir)

        ### ------------- MLP -----------------
        # gate proj
        w_name = f"{prefix}.mlp.gate_proj.weight"
        weight_t = sd[w_name]

        layer_tag = f"layer_{i:02d}_gate_proj"
        save_linear_quantized(layer_tag, weight_t, mistral_decoder_layers_quant_dir)

        # up proj
        w_name = f"{prefix}.mlp.up_proj.weight"
        weight_t = sd[w_name]

        layer_tag = f"layer_{i:02d}_up_proj"
        save_linear_quantized(layer_tag, weight_t, mistral_decoder_layers_quant_dir)

        # down proj
        w_name = f"{prefix}.mlp.down_proj.weight"
        weight_t = sd[w_name]

        layer_tag = f"layer_{i:02d}_down_proj"
        save_linear_quantized(layer_tag, weight_t, mistral_decoder_layers_quant_dir)

        ### ------------- RMSNorm (No quantization) -----------------
        # input layernorm
        w_name = f"{prefix}.input_layernorm.weight"
        weight_t = sd[w_name]
        w_np = weight_t.detach().to("cpu").numpy()

        layer_tag = f"layer_{i:02d}_input_layernorm"
        np.save(
            os.path.join(mistral_decoder_layers_quant_dir, f"{layer_tag}.npy"), w_np
        )

        # post attention layernorm
        w_name = f"{prefix}.post_attention_layernorm.weight"
        weight_t = sd[w_name]
        w_np = weight_t.detach().to("cpu").numpy()

        layer_tag = f"layer_{i:02d}_post_attention_layernorm"
        np.save(
            os.path.join(mistral_decoder_layers_quant_dir, f"{layer_tag}.npy"), w_np
        )

    weight_embed = sd[f"model.embed_tokens.weight"]
    weight_norm = sd[f"model.norm.weight"]
    weight_lm_head = sd[f"lm_head.weight"]

    save_linear(
        norm_t=weight_norm,
        embed_t=weight_embed,
        head_t=weight_lm_head,
        filepath=mistral_other_layers_quant_path,
    )

    print("Done.")


if __name__ == "__main__":
    main()
