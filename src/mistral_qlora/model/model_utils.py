import math
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from mistral_qlora.config import (
    LoRA_r,
    alpha,
    dropout,
    embed_dim,
    head_dim,
    hidden_size_atten,
    hidden_size_mlp,
    lora_true,
    num_attention_heads,
    num_key_value_heads,
    num_layers,
    rms_norm_eps,
    rope_theta,
    vocab_size,
)
from mistral_qlora.quant.utils_linear import Linear, LoRALinear, QuantizedLinear

LORA_TARGETS = ("q", "k", "v", "o", "gate", "up", "down")


def resolve_use_lora(use_lora: "dict | bool") -> dict:
    """Normalize the `bool` shorthand into the per-projection dict.

    `True`/`False` applies to every projection in LORA_TARGETS; a dict selects
    them individually and passes through unchanged.
    """
    if isinstance(use_lora, bool):
        return dict.fromkeys(LORA_TARGETS, use_lora)
    return use_lora


@dataclass
class MistralConfig:
    vocab_size: int = vocab_size
    embed_dim: int = embed_dim
    alpha: float = alpha
    dropout: float = dropout
    r: int = LoRA_r
    lora_true: dict = field(default_factory=lambda: lora_true.copy())
    hidden_size_atten: int = hidden_size_atten
    rms_norm_eps: float = rms_norm_eps
    num_attention_heads: int = num_attention_heads
    num_key_value_heads: int = num_key_value_heads
    head_dim: int = head_dim
    rope_theta: float = rope_theta
    hidden_size_mlp: int = hidden_size_mlp
    num_layers: int = num_layers


class MistralAttention(nn.Module):
    """
    Implements MistralAttention module in MLX, respecting original configuration and structure.
    Linear Layers, QuantizedLayers and LoraLayers are all supported.
    """

    def __init__(
        self, config: MistralConfig, *, linear_cls=Linear, use_bias: bool = False
    ):
        super().__init__()

        self.hidden_size = config.hidden_size_atten
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

        self.r = config.r
        self.alpha = config.alpha
        self.dropout = config.dropout

        self.inv_freq = 1.0 / (
            self.rope_theta
            ** (mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim)
        )

        assert self.hidden_size % self.num_heads == 0, (
            "hidden_size must be divisible by num_attention_heads"
        )

        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim

        self.q_proj = linear_cls(self.hidden_size, q_out, bias=use_bias)
        self.k_proj = linear_cls(self.hidden_size, kv_out, bias=use_bias)
        self.v_proj = linear_cls(self.hidden_size, kv_out, bias=use_bias)
        self.o_proj = linear_cls(self.hidden_size, self.hidden_size, bias=use_bias)

    @classmethod
    def from_quantized_weights(cls, config: MistralConfig, packed_weights: dict):
        """
        Returns a MistralAttention initialized with provided weights & config.
        Inputs:
        - config, MistralConfig
        - packed_weights, dict, looks like :  {
                                        "q_proj": {
                                            "weight_q": quantized_weights,
                                            "scale": scale,
                                            "row_min": row_min,
                                            "orig_in": orig_in
                                            },
                                        "k_proj": {
                                            "weight_q": quantized_weights,
                                            "scale": scale,
                                            "row_min": row_min,
                                            "orig_in": orig_in
                                            },
                                        ...
                                    }
        """
        attn = cls(config)
        r = attn.r
        alpha = attn.alpha
        dropout = attn.dropout

        packed_weights_q = packed_weights["q_proj"]
        attn.q_proj = QuantizedLinear.from_packed(
            packed_weights_q["weight_q"],
            packed_weights_q["scale"],
            packed_weights_q["row_min"],
            packed_weights_q["orig_in"],
        )
        if config.lora_true["q"]:
            attn.q_proj = LoRALinear.from_quantLinear(
                base=attn.q_proj, r=r, alpha=alpha, dropout=dropout
            )
        packed_weights_v = packed_weights["v_proj"]
        attn.v_proj = QuantizedLinear.from_packed(
            packed_weights_v["weight_q"],
            packed_weights_v["scale"],
            packed_weights_v["row_min"],
            packed_weights_v["orig_in"],
        )
        if config.lora_true["v"]:
            attn.v_proj = LoRALinear.from_quantLinear(
                base=attn.v_proj, r=r, alpha=alpha, dropout=dropout
            )

        packed_weights_k = packed_weights["k_proj"]
        attn.k_proj = QuantizedLinear.from_packed(
            packed_weights_k["weight_q"],
            packed_weights_k["scale"],
            packed_weights_k["row_min"],
            packed_weights_k["orig_in"],
        )
        if config.lora_true["k"]:
            attn.k_proj = LoRALinear.from_quantLinear(
                base=attn.k_proj, r=r, alpha=alpha, dropout=dropout
            )

        packed_weights_o = packed_weights["o_proj"]
        attn.o_proj = QuantizedLinear.from_packed(
            packed_weights_o["weight_q"],
            packed_weights_o["scale"],
            packed_weights_o["row_min"],
            packed_weights_o["orig_in"],
        )
        if config.lora_true["o"]:
            attn.o_proj = LoRALinear.from_quantLinear(
                base=attn.o_proj, r=r, alpha=alpha, dropout=dropout
            )

        return attn

    @classmethod
    def from_weights(cls, config, weights: dict):
        """
        Given weights dict, initialize the MistralAttention
        class with LoRALinear layers with given weights.
        Inputs:
                - config: MistralConfig
                - weights: dict of weigths
                    (e.g, weights["q_proj"] returns mx.array
                    representing weights of the q projection)

        Calls QuantizedLinear on weights then wraps it using LoRALinear
        """
        attn = cls(config)
        r = attn.r
        alpha = attn.alpha
        dropout = attn.dropout

        base_q = QuantizedLinear.convert_4bit(weights["q_proj"], None)
        attn.q_proj = LoRALinear(base=base_q, r=r, alpha=alpha, dropout=dropout)

        base_k = QuantizedLinear.convert_4bit(weights["k_proj"], None)
        attn.k_proj = LoRALinear(base=base_k, r=r, alpha=alpha, dropout=dropout)

        base_v = QuantizedLinear.convert_4bit(weights["v_proj"], None)
        attn.v_proj = LoRALinear(base=base_v, r=r, alpha=alpha, dropout=dropout)

        base_o = QuantizedLinear.convert_4bit(weights["o_proj"], None)
        attn.o_proj = LoRALinear(base=base_o, r=r, alpha=alpha, dropout=dropout)

        return attn

    def _shape_q(self, x):
        """
        (B, T, H*Dh) -> (B, num_heads, T, head_dim)
        Converts q output,
        adding another dimension; splits Heads_q*Head_Dim dimension.
        """
        B, T, _ = x.shape
        x = x.reshape(B, T, self.num_heads, self.head_dim)
        x = mx.transpose(x, (0, 2, 1, 3))
        return x

    def _shape_kv(self, x):
        """
        (B, T, H_kv*Dh) -> (B, num_kv_heads, T, head_dim)
         Converts q output,
        adding another dimension; splits Heads_kv*Head_Dim dimension.
        """
        B, T, _ = x.shape
        x = x.reshape(B, T, self.num_kv_heads, self.head_dim)
        x = mx.transpose(x, (0, 2, 1, 3))
        return x

    def _expand_kv(self, k, v):
        """
        Expand KV heads from num_kv_heads to num_heads:
            k: (B, H_kv, T, Dh)
            v: (B, H_kv, T, Dh)
        Return:
            k, v: (B, H_q, T, Dh)
        """
        if self.num_kv_heads == self.num_heads:
            return k, v

        assert self.num_heads % self.num_kv_heads == 0, (
            "num_heads must be multiple of num_key_value_heads"
        )

        repeat = self.num_heads // self.num_kv_heads
        k = mx.repeat(k, repeat, axis=1)
        v = mx.repeat(v, repeat, axis=1)
        return k, v

    def _apply_rope(self, x, positions):
        """
        Apply rotary position embeddings to x.
        x: (B, H, T, Dh)
        positions: (T,) T being sequence length

        Returns: x_rot (same shape as x)
        """

        freqs = mx.outer(positions.astype(mx.float32), self.inv_freq)
        cos = mx.cos(freqs)[None, None, :, :]
        sin = mx.sin(freqs)[None, None, :, :]

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x_rot_first = x1 * cos - x2 * sin
        x_rot_second = x1 * sin + x2 * cos

        x_rot = mx.concatenate([x_rot_first, x_rot_second], axis=-1)
        return x_rot

    def __call__(
        self,
        x: mx.array,
        *,
        attn_mask: mx.array | None = None,
        cache: dict | None = None,
        positions: mx.array | None = None,
        use_lora: dict | bool = False,
    ):
        """
        x: (B, T, D)
        use_lora:
            dict, in the form of {"v": True, "k": True, "o": False, "q": False}.
        attn_mask:
            Optional, broadcastable to (B, 1, T, S),
            contains 0 for allowed, -inf for masked.
        cache:
            Optional dict with "k" and "v" for K & V cache:
                "k": (B, H_kv, T_past, Dh)
                "v": (B, H_kv, T_past, Dh)
        positions:
            Optional (T,) array of positions for RoPE.
            If None, uses range with offset from cache length.
        """
        B, T, _ = x.shape
        use_lora = resolve_use_lora(use_lora)

        q = self.q_proj(x, use_lora=use_lora["q"])
        k = self.k_proj(x, use_lora=use_lora["k"])
        v = self.v_proj(x, use_lora=use_lora["v"])

        q = self._shape_q(q)
        k = self._shape_kv(k)
        v = self._shape_kv(v)

        if cache is not None and "k" in cache:
            past_len = cache["k"].shape[2]
        else:
            past_len = 0
        if positions is None:
            positions = mx.arange(past_len, past_len + T, dtype=mx.int32)

        q = self._apply_rope(q, positions)
        k = self._apply_rope(k, positions)

        if cache is not None and "k" in cache and "v" in cache:
            k = mx.concatenate([cache["k"], k], axis=2)
            v = mx.concatenate([cache["v"], v], axis=2)

        new_cache = {"k": k, "v": v}

        k, v = self._expand_kv(k, v)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale

        if attn_mask is not None:
            scores = scores + attn_mask

        attn_weights = mx.softmax(scores, axis=-1)

        context = mx.matmul(attn_weights, v)
        context = mx.transpose(context, (0, 2, 1, 3))
        context = context.reshape(B, T, self.hidden_size)

        out = self.o_proj(context, use_lora=use_lora["o"])

        return out, new_cache


class MistralMLP(nn.Module):
    def __init__(
        self, config: MistralConfig, *, linear_cls=Linear, use_bias: bool = False
    ):
        super().__init__()

        self.hidden_size = config.hidden_size_mlp
        self.input_size = config.hidden_size_atten
        self.r = config.r
        self.alpha = config.alpha
        self.dropout = config.dropout

        self.gate_proj = linear_cls(self.input_size, self.hidden_size, bias=use_bias)
        self.up_proj = linear_cls(self.input_size, self.hidden_size, bias=use_bias)
        self.down_proj = linear_cls(self.hidden_size, self.input_size, bias=use_bias)

    @classmethod
    def from_quantized_weights(cls, config: MistralConfig, packed_weights: dict):
        """
        Returns a MistralMLP initialized with provided weights & config.
        Inputs:
        - config, MistralConfig
        - packed_weights, dict, looks like :  {
                                        "gate_proj": {
                                            "weight_q": quantized_weights,
                                            "scale": scale,
                                            "row_min": row_min,
                                            "orig_in": orig_in
                                            },
                                        "down_proj": {
                                            "weight_q": quantized_weights,
                                            "scale": scale,
                                            "row_min": row_min,
                                            "orig_in": orig_in
                                            },
                                        ...
                                    }
        """
        mlp = cls(config)
        r = mlp.r
        alpha = mlp.alpha
        dropout = mlp.dropout

        packed_weights_gate = packed_weights["gate_proj"]
        mlp.gate_proj = QuantizedLinear.from_packed(
            packed_weights_gate["weight_q"],
            packed_weights_gate["scale"],
            packed_weights_gate["row_min"],
            packed_weights_gate["orig_in"],
        )
        if config.lora_true["gate"]:
            mlp.gate_proj = LoRALinear.from_quantLinear(
                base=mlp.gate_proj, r=r, alpha=alpha, dropout=dropout
            )

        packed_weights_down = packed_weights["down_proj"]
        mlp.down_proj = QuantizedLinear.from_packed(
            packed_weights_down["weight_q"],
            packed_weights_down["scale"],
            packed_weights_down["row_min"],
            packed_weights_down["orig_in"],
        )
        if config.lora_true["down"]:
            mlp.down_proj = LoRALinear.from_quantLinear(
                base=mlp.down_proj, r=r, alpha=alpha, dropout=dropout
            )

        packed_weights_up = packed_weights["up_proj"]
        mlp.up_proj = QuantizedLinear.from_packed(
            packed_weights_up["weight_q"],
            packed_weights_up["scale"],
            packed_weights_up["row_min"],
            packed_weights_up["orig_in"],
        )
        if config.lora_true["up"]:
            mlp.up_proj = LoRALinear.from_quantLinear(
                base=mlp.up_proj, r=r, alpha=alpha, dropout=dropout
            )

        return mlp

    @classmethod
    def from_weights(cls, config, weights: dict):
        """
        Given weights dict, initialize the MistralMLP class with LoRALinear layers.

        Inputs:
                - config: MistralConfig
                - weights: dict of weigths
                    (e.g, weights["down_proj"] returns mx.array
                    representing weights of the down projection)

        Calls QuantizedLinear on weights then wraps it using LoRALinear
        """
        mlp = cls(config)
        r = mlp.r
        alpha = mlp.alpha
        dropout = mlp.dropout

        base_q = QuantizedLinear.convert_4bit(weights["gate_proj"], None)
        mlp.gate_proj = LoRALinear(base=base_q, r=r, alpha=alpha, dropout=dropout)

        base_k = QuantizedLinear.convert_4bit(weights["up_proj"], None)
        mlp.up_proj = LoRALinear(base=base_k, r=r, alpha=alpha, dropout=dropout)

        base_v = QuantizedLinear.convert_4bit(weights["down_proj"], None)
        mlp.down_proj = LoRALinear(base=base_v, r=r, alpha=alpha, dropout=dropout)

        return mlp

    def __call__(
        self,
        x: mx.array,
        *,
        use_lora: dict | bool = False,
    ):
        """
        Forward pass in the MLP block given an input x: mx.array.
        Handles Linear, QuantizedLinear and LoRALinear layers uniformly.
        """
        use_lora = resolve_use_lora(use_lora)

        gate = self.gate_proj(x, use_lora=use_lora["gate"])
        up = self.up_proj(x, use_lora=use_lora["up"])

        h = nn.silu(gate) * up
        out = self.down_proj(h, use_lora=use_lora["down"])

        return out
