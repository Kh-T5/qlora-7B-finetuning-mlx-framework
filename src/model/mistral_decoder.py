import mlx.nn as nn
from src.model.model_utils import MistralMLP, MistralAttention, MistralConfig
import mlx.core as mx
from src.config import mistral_other_layers_quant_path
import os
import numpy as np


class MistralDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MistralConfig,
        *,
        attn_block=MistralAttention,
        mlp_block=MistralMLP,
        linear_cls=nn.Linear,
    ):
        super().__init__()

        h_dim = config.hidden_size_atten
        eps = config.rms_norm_eps

        # RMSNorm layers
        self.input_layernorm = nn.RMSNorm(h_dim, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(h_dim, eps=eps)

        # MLP & Attention
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

        # New class
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
        # Proj names
        names_attn = ["v_proj", "k_proj", "q_proj", "o_proj"]
        names_mlp = ["gate_proj", "down_proj", "up_proj"]

        # Split weigths dict
        weights_mlp = {name: weights[name] for name in names_mlp}
        weights_attn = {name: weights[name] for name in names_attn}

        # New class
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

        # Attention block
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

        # MLP block
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
        linear_cls=nn.Linear,
    ):
        super().__init__()

        self.num_layers = config.num_layers
        self.layers = [
            decoder_layer(config, attn_block=attn, mlp_block=mlp, linear_cls=linear_cls)
            for _ in range(self.num_layers)
        ]

        self.final_norm = nn.RMSNorm(config.hidden_size_atten, eps=config.rms_norm_eps)

    @classmethod
    def build_decoder_from_npz(cls, config: MistralConfig, dir: str):
        """
        Constructor for MistralDecoder block: 32 * MistralDecoderLayer,
        Mirors original Mistral architechure:
                                        (0-31): 32 x MistralDecoderLayer(
                                    (self_attn): MistralAttention(
                                    (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
                                    (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
                                    (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
                                    (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
                                    )
                                    (mlp): MistralMLP(
                                    (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
                                    (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
                                    (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
                                    (act_fn): SiLUActivation()
                                    )
        Inputs:
            - config, MistralConfig
            - dir, str, directory where pre-trained model weigths are stored

        """
        names_attn = ["v_proj", "k_proj", "q_proj", "o_proj"]
        names_mlp = ["gate_proj", "down_proj", "up_proj"]
        names_norm = ["input", "post_attention"]
        sfx_quant = ".npz"
        sfx_no_quant = ".npy"
        new_decoder = cls(config)
        new_decoder.layers = []

        for i in range(config.num_layers):

            # Load attention quantized weights
            packed_weights_attn = {}
            for name in names_attn:
                path = os.path.join(dir, f"layer_{i:02d}_{name}{sfx_quant}")
                with np.load(path) as data:
                    packed_weights_attn[name] = {
                        "weight_q": data["weight_q"],
                        "scale": data["scale"],
                        "row_min": data["row_min"],
                        "orig_in": int(data["orig_in"]),
                    }

            # Load mlp quantized weights
            packed_weights_mlp = {}
            for name in names_mlp:
                path = os.path.join(dir, f"layer_{i:02d}_{name}{sfx_quant}")
                with np.load(path) as data:
                    packed_weights_mlp[name] = {
                        "weight_q": data["weight_q"],
                        "scale": data["scale"],
                        "row_min": data["row_min"],
                        "orig_in": int(data["orig_in"]),
                    }

            # Load RMSNorm weights
            weights_norm = {}
            for name in names_norm:
                path = os.path.join(
                    dir, f"layer_{i:02d}_{name}_layernorm{sfx_no_quant}"
                )
                weights_norm[name] = mx.array(np.load(path))

            new_decoder.layers.append(
                MistralDecoderLayer.from_quantized_weights(
                    config,
                    packed_weights_mlp=packed_weights_mlp,
                    packed_weights_attn=packed_weights_attn,
                    weights_norm=weights_norm,
                )
            )
            with np.load(mistral_other_layers_quant_path) as data:
                new_decoder.final_norm.weight = mx.array(data["norm_np"])
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

        for layer, layer_cache in zip(self.layers, caches):
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
