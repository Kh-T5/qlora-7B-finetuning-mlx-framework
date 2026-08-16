import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mistral_qlora.config import MistralConfig
from mistral_qlora.constants import (
    ATTN_PROJECTIONS,
    LAYER_NORM_TEMPLATE,
    LAYER_WEIGHT_TEMPLATE,
    MLP_PROJECTIONS,
    NORM_NAMES,
)
from mistral_qlora.model.model_utils import MistralAttention, MistralMLP
from mistral_qlora.quant.utils_linear import Linear


def _load_packed(layers_dir: str, index: int, name: str) -> dict:
    """Read one quantized projection into the dict `from_packed` expects."""
    path = os.path.join(
        layers_dir, LAYER_WEIGHT_TEMPLATE.format(index=index, name=name)
    )
    with np.load(path) as data:
        return {
            "weight_q": data["weight_q"],
            "scale": data["scale"],
            "row_min": data["row_min"],
            "orig_in": int(data["orig_in"]),
        }


def _load_norm(layers_dir: str, index: int, name: str) -> mx.array:
    """Read one unquantized RMSNorm weight."""
    path = os.path.join(layers_dir, LAYER_NORM_TEMPLATE.format(index=index, name=name))
    return mx.array(np.load(path), dtype=mx.float16)


class MistralDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MistralConfig,
        *,
        attn_block=MistralAttention,
        mlp_block=MistralMLP,
        linear_cls=Linear,
    ):
        super().__init__()

        h_dim = config.hidden_size_atten
        eps = config.rms_norm_eps

        self.input_layernorm = nn.RMSNorm(h_dim, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(h_dim, eps=eps)

        self.attn = attn_block(config, linear_cls=linear_cls)
        self.mlp = mlp_block(config, linear_cls=linear_cls)

    @classmethod
    def from_quantized_weights(
        cls,
        config: MistralConfig,
        packed_weights_mlp: dict,
        packed_weights_attn: dict,
        weights_norm: dict,
    ):
        """
        Inputs:
            - config, MistralConfig
            - packed_weights_mlp, dict (cf. MistralMLP.from_quantized_weights)
            - packed_weights_attn, dict (cf. MistralAttention.from_quantized_weights)

        Returns MistralDecoderLayer object with saved weights,
                handles LoRALinear, QuantizedLinear and nn.Linear
        """

        decoder = cls(config)
        decoder.attn = MistralAttention.from_quantized_weights(
            config, packed_weights_attn
        )
        decoder.mlp = MistralMLP.from_quantized_weights(config, packed_weights_mlp)
        decoder.input_layernorm.weight = weights_norm["input"]
        decoder.post_attention_layernorm.weight = weights_norm["post_attention"]

        return decoder

    @classmethod
    def from_weights(cls, config: MistralConfig, weights: dict):
        """
        Inputs:
            - config : MistralConfig
            - weights : dict, contains saved weights of the pre-trained model

        Returns MistralDecoderLayer object with saved weights,
                handles LoRALinear, QuantizedLinear and nn.Linear
        """
        names_attn = ["v_proj", "k_proj", "q_proj", "o_proj"]
        names_mlp = ["gate_proj", "down_proj", "up_proj"]

        weights_mlp = {name: weights[name] for name in names_mlp}
        weights_attn = {name: weights[name] for name in names_attn}

        decoder = cls(config)
        decoder.attn = MistralAttention.from_weights(config, weights_attn)
        decoder.mlp = MistralMLP.from_weights(config, weights_mlp)
        decoder.input_layernorm.weight = weights["input"]
        decoder.post_attention_layernorm.weight = weights["post_attention"]

        return decoder

    def __call__(
        self,
        x: mx.array,
        *,
        attn_mask: mx.array | None = None,
        positions: mx.array | None = None,
        cache: dict | None = None,
        use_lora: dict | bool = False,
    ):
        """
        Forward pass: Mirors forward pass of Hugging Face Mistral-7B
        x -> rms -> attention -> add residual -> rms -> mlp -> output

        - Keeps track of cache (k, v proj in self attention)
        - attn_mask, = 0 | -inf, applied before softmax in attention head
        - positions, called for RoPE
        """

        residual = x
        h = self.input_layernorm(x)
        h, new_cache = self.attn(
            h,
            attn_mask=attn_mask,
            cache=cache,
            positions=positions,
            use_lora=use_lora,
        )
        x = residual + h

        residual = x
        h = self.post_attention_layernorm(x)
        h = self.mlp(h, use_lora=use_lora)
        x = residual + h

        return x, new_cache


class MistralDecoder(nn.Module):
    def __init__(
        self,
        config: MistralConfig,
        *,
        decoder_layer=MistralDecoderLayer,
        attn=MistralAttention,
        mlp=MistralMLP,
        linear_cls=Linear,
    ):
        super().__init__()

        self.num_layers = config.num_layers
        self.layers = [
            decoder_layer(config, attn_block=attn, mlp_block=mlp, linear_cls=linear_cls)
            for _ in range(self.num_layers)
        ]

        self.final_norm = nn.RMSNorm(config.hidden_size_atten, eps=config.rms_norm_eps)

    @classmethod
    def build_decoder_from_npz(
        cls, config: MistralConfig, layers_dir: str, norm_path: str
    ):
        """Build the decoder stack from a quantized checkpoint on disk.

        Inputs:
            config: MistralConfig
            layers_dir: directory holding the per-layer .npz and .npy files
            norm_path: .npz holding the final RMSNorm weight under "norm_np"
        """
        new_decoder = cls(config)
        new_decoder.layers = []

        for i in range(config.num_layers):
            packed = {
                name: _load_packed(layers_dir, i, name)
                for name in ATTN_PROJECTIONS + MLP_PROJECTIONS
            }
            weights_norm = {
                name: _load_norm(layers_dir, i, name) for name in NORM_NAMES
            }

            new_decoder.layers.append(
                MistralDecoderLayer.from_quantized_weights(
                    config,
                    packed_weights_mlp={n: packed[n] for n in MLP_PROJECTIONS},
                    packed_weights_attn={n: packed[n] for n in ATTN_PROJECTIONS},
                    weights_norm=weights_norm,
                )
            )

        with np.load(norm_path) as data:
            new_decoder.final_norm.weight = mx.array(data["norm_np"], dtype=mx.float16)

        return new_decoder

    def __call__(
        self,
        x: mx.array,
        *,
        attn_mask: mx.array | None = None,
        caches: list[dict] | None = None,
        positions: mx.array | None = None,
        use_lora: dict | bool = False,
    ):
        if caches is None:
            caches = [None] * self.num_layers

        new_caches = []

        for layer, layer_cache in zip(self.layers, caches, strict=True):
            x, new_cache = layer(
                x,
                attn_mask=attn_mask,
                cache=layer_cache,
                positions=positions,
                use_lora=use_lora,
            )
            new_caches.append(new_cache)

        x = self.final_norm(x)
        return x, new_caches
