import mlx.nn as nn
from src.config import *
from dataclasses import dataclass
from src.quant.utils_linear import LoRALinear, QuantizedLinear
import mlx.core as mx
import math


@dataclass
class MistralConfig:
    ### Lora
    alpha: float
    dropout: float
    r: int = 16
    ### Attention
    hidden_size: int = 4096
    rms_norm_eps: float = 1e-5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 1e4


### ----------------- Attention Block -----------------------
class MistralAttention(nn.Module):
    """
    Implements MistralAttention module in MLX, respecting original configuration and structure.
    Linear Layers, QuantizedLayers and LoraLayers are all supported.
    """

    def __init__(
        self, config: MistralConfig, *, linear_cls=nn.Linear, use_bias: bool = False
    ):
        super().__init__()

        # Attention params
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

        # LoRA params
        self.r = config.r
        self.alpha = config.alpha
        self.dropout = config.dropout

        # Rotation
        self.inv_freq = 1.0 / (
            self.rope_theta
            ** (mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim)
        )

        # Quick check
        assert (
            self.hidden_size % self.num_heads == 0
        ), "hidden_size must be divisible by num_attention_heads"

        # projections set up
        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim

        self.q_proj = linear_cls(self.hidden_size, q_out, bias=use_bias)
        self.k_proj = linear_cls(self.hidden_size, kv_out, bias=use_bias)
        self.v_proj = linear_cls(self.hidden_size, kv_out, bias=use_bias)
        self.o_proj = linear_cls(self.hidden_size, self.hidden_size, bias=use_bias)

    @classmethod
    def from_quantized_weights(cls, config, weights: dict):
        """
        Given weights dict, initialize the MistralAttention class with LoRALinear layers.

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

    def _lora_or_linear(self, layer, x, use_lora: bool):
        """
        Runs the forward pass for a plain Linear/QuantizedLinear layer or with LoRALinear
        """
        try:
            return layer(x, use_lora=use_lora)  # LoRA layer
        except TypeError:
            return layer(x)  # Linear or QuantizedLinear layer

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

        assert (
            self.num_heads % self.num_kv_heads == 0
        ), "num_heads must be multiple of num_key_value_heads"

        repeat = self.num_heads // self.num_kv_heads
        # repeat along head dimension
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

        # x: (B, H, T, Dh)
        x_ = mx.transpose(x, (0, 1, 2, 3))  # just alias

        x1 = x_[..., ::2]
        x2 = x_[..., 1::2]

        # RoPE rotation
        x_rot_first = x1 * cos - x2 * sin
        x_rot_second = x1 * sin + x2 * cos

        x_rot = mx.concatenate([x_rot_first, x_rot_second], axis=-1)
        return x_rot

        ## -------------- Forward Pass ---------------------

    def __call__(
        self,
        x: mx.array,
        *,
        attn_mask: mx.array | None = None,
        cache: dict | None = None,
        positions: mx.array | None = None,
        use_lora: bool = True,
    ):
        """
        x: (B, T, D)
        attn_mask:
            Optional, broadcastable to (B, 1, T, S),
            usually contains 0 for allowed, -inf for masked.
        cache:
            Optional dict with "k" and "v" for KV cache:
                "k": (B, H_kv, T_past, Dh)
                "v": (B, H_kv, T_past, Dh)
        positions:
            Optional (T,) array of positions for RoPE.
            If None, uses range with offset from cache length.
        use_lora:
            If True and layers are LoRALinear, add LoRA adapters.
        """
        B, T, _ = x.shape

        # Linear proj
        q = self._lora_or_linear(self.q_proj, x, use_lora=use_lora)
        k = self._lora_or_linear(self.k_proj, x, use_lora=use_lora)
        v = self._lora_or_linear(self.v_proj, x, use_lora=use_lora)

        # Reshape intop heads for attention operations:
        q = self._shape_q(q)
        k = self._shape_kv(k)
        v = self._shape_kv(v)

        # RoPE
        if cache is not None and "k" in cache:
            past_len = cache["k"].shape[2]
        else:
            past_len = 0
        if positions is None:
            positions = mx.arange(past_len, past_len + T, dtype=mx.int32)

        q = self._apply_rope(q, positions)
        k = self._apply_rope(k, positions)

        # Update cache
        if cache is not None and "k" in cache and "v" in cache:
            k = mx.concatenate([cache["k"], k], axis=2)
            v = mx.concatenate([cache["v"], v], axis=2)

        # Keeps track of k, v for future tokens before expanding
        new_cache = {"k": k, "v": v}

        # Expansion of k, v heads
        k, v = self._expand_kv(k, v)  # (B, H_q, S, Dh)
        S = k.shape[2]

        # Attention mechanism
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale

        if attn_mask is not None:
            # mask is 0 or -inf
            scores = scores + attn_mask

        attn_weights = mx.softmax(scores, axis=-1)

        # Output
        context = mx.matmul(attn_weights, v)
        context = mx.transpose(context, (0, 2, 1, 3))
        context = context.reshape(B, T, self.hidden_size)

        out = self._lora_or_linear(self.o_proj, context, use_lora)

        return out, new_cache


### ----------------- MLP Block -----------------------
