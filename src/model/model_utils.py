import mlx.nn as nn
from dataclasses import dataclass
from src.quant.utils_linear import LoRALinear, QuantizedLinear
import mlx.core as mx
import math
from src.config import (
    alpha,
    dropout,
    LoRA_r,
    hidden_size_atten,
    rms_norm_eps,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    rope_theta,
    hidden_size_mlp,
    num_layers,
    vocab_size,
    embed_dim,
    lora_true,
)


@dataclass
class MistralConfig:
    # Embedding
    vocab_size: int = vocab_size
    embed_dim: int = embed_dim
    ### LoRA
    alpha: float = alpha
    dropout: float = dropout
    r: int = LoRA_r
    lora_true: dict = lora_true
    ### Attention
    hidden_size_atten: int = hidden_size_atten
    rms_norm_eps: float = rms_norm_eps
    num_attention_heads: int = num_attention_heads
    num_key_value_heads: int = num_key_value_heads
    head_dim: int = head_dim
    rope_theta: float = rope_theta
    ### MLP
    hidden_size_mlp: int = hidden_size_mlp
    ### Decoder
    num_layers: int = num_layers


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
        self.hidden_size = config.hidden_size_atten
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

        # q_proj
        packed_weights_q = packed_weights["q_proj"]
        base_q = QuantizedLinear.from_packed(
            packed_weights_q["weight_q"],
            packed_weights_q["scale"],
            packed_weights_q["row_min"],
            packed_weights_q["orig_in"],
        )
        attn.q_proj = LoRALinear(base=base_q, r=r, alpha=alpha, dropout=dropout)

        # v_proj
        packed_weights_v = packed_weights["v_proj"]
        base_v = QuantizedLinear.from_packed(
            packed_weights_v["weight_q"],
            packed_weights_v["scale"],
            packed_weights_v["row_min"],
            packed_weights_v["orig_in"],
        )
        attn.v_proj = LoRALinear(base=base_v, r=r, alpha=alpha, dropout=dropout)

        # k_proj
        packed_weights_k = packed_weights["k_proj"]
        base_k = QuantizedLinear.from_packed(
            packed_weights_k["weight_q"],
            packed_weights_k["scale"],
            packed_weights_k["row_min"],
            packed_weights_k["orig_in"],
        )
        attn.k_proj = LoRALinear(base=base_k, r=r, alpha=alpha, dropout=dropout)

        # o_proj
        packed_weights_o = packed_weights["o_proj"]
        base_o = QuantizedLinear.from_packed(
            packed_weights_o["weight_q"],
            packed_weights_o["scale"],
            packed_weights_o["row_min"],
            packed_weights_o["orig_in"],
        )
        attn.o_proj = LoRALinear(base=base_o, r=r, alpha=alpha, dropout=dropout)

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

        # Linear proj
        q = self._lora_or_linear(self.q_proj, x, use_lora=use_lora["q"])
        k = self._lora_or_linear(self.k_proj, x, use_lora=use_lora["k"])
        v = self._lora_or_linear(self.v_proj, x, use_lora=use_lora["v"])

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
        k, v = self._expand_kv(k, v)

        # Attention mechanism [ Softmax(Q@K.T/scale ) ]
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

        out = self._lora_or_linear(self.o_proj, context, use_lora["o"])

        return out, new_cache


### ----------------- MLP Block -----------------------


class MistralMLP(nn.Module):
    def __init__(
        self, config: MistralConfig, *, linear_cls=nn.Linear, use_bias: bool = False
    ):
        super().__init__()

        # Attention params
        self.hidden_size = config.hidden_size_mlp
        self.input_size = config.hidden_size_atten
        # LoRA params
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

        # gate_proj
        packed_weights_gate = packed_weights["gate_proj"]
        base_gate = QuantizedLinear.from_packed(
            packed_weights_gate["weight_q"],
            packed_weights_gate["scale"],
            packed_weights_gate["row_min"],
            packed_weights_gate["orig_in"],
        )
        mlp.gate_proj = LoRALinear(base=base_gate, r=r, alpha=alpha, dropout=dropout)

        # down_proj
        packed_weights_down = packed_weights["down_proj"]
        base_down = QuantizedLinear.from_packed(
            packed_weights_down["weight_q"],
            packed_weights_down["scale"],
            packed_weights_down["row_min"],
            packed_weights_down["orig_in"],
        )
        mlp.down_proj = LoRALinear(base=base_down, r=r, alpha=alpha, dropout=dropout)

        # up_proj
        packed_weights_up = packed_weights["up_proj"]
        base_up = QuantizedLinear.from_packed(
            packed_weights_up["weight_q"],
            packed_weights_up["scale"],
            packed_weights_up["row_min"],
            packed_weights_up["orig_in"],
        )
        mlp.up_proj = LoRALinear(base=base_up, r=r, alpha=alpha, dropout=dropout)

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

    def _lora_or_linear(self, layer, x, use_lora: bool):
        """
        Runs the forward pass for a plain Linear/QuantizedLinear layer or with LoRALinear
        """
        try:
            return layer(x, use_lora=use_lora)  # LoRA layer
        except TypeError:
            return layer(x)  # nn.Linear or QuantizedLinear layer

    def __call__(
        self,
        x: mx.array,
        *,
        use_lora: dict | bool = False,
    ):
        """
        Forward pass in the MLP block given an input x: mx.array.
        Handles nn.Linear, QuantizationLinear and LoRALinear layers.
        """

        gate = self._lora_or_linear(self.gate_proj, x, use_lora=use_lora["gate"])
        up = self._lora_or_linear(self.up_proj, x, use_lora=use_lora["up"])

        h = nn.silu(gate) * up
        out = self._lora_or_linear(self.down_proj, h, use_lora=use_lora["down"])

        return out
