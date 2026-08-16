import mlx.core as mx
import mlx.nn as nn

from mistral_qlora.quant.quant_4bit import (
    dequantize_4bit_per_row,
    quantize_4bit_per_row,
)


class Linear(nn.Linear):
    """`nn.Linear` carrying the shared projection signature.

    Accepts `use_lora` and ignores it, so a projection can be called without the
    caller knowing whether it holds adapters.
    """

    def __call__(self, x: mx.array, use_lora: bool = False) -> mx.array:
        return super().__call__(x)


class QuantizedLinear(nn.Module):
    """
    Linear layer corresponding to frozen 4bit mx.arrays.
    It performs dequantization internally in the forward pass.

    Internally stores:
        - weight_q: (out_features x ceil(in_features/2)) uint8
        - scale:    (out_features) float32
        - row_min:  (out_features) float32
        - orig_in_features: int

    Forward:
        x: (batch x in_features)
        y = x @ W^T (+ bias), where W is dequantized on forward pass only.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        dtype=mx.float16,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        packed_cols = (in_features + 1) // 2

        self.quant_W = mx.zeros((out_features, packed_cols), dtype=mx.uint8)
        self.scale = mx.ones((out_features,), dtype=dtype)
        self.row_min = mx.zeros((out_features,), dtype=dtype)
        self.orig_in_features = in_features

        if bias:
            self.bias = mx.zeros((out_features,), dtype=dtype)
        else:
            self.bias = None

    @classmethod
    def convert_4bit(cls, weight: mx.array, bias=None) -> "QuantizedLinear":
        """
        Convenience constructor from a full-precision weight matrix.

        Inputs:
            weight: (out_features x in_features) float array
            bias:   (out_features) float mlx array

        Returns:
            QuantizedLinear with 4-bit weights corresponding to the frozen "weight".
        """
        out_features, in_features = weight.shape
        new_cls = cls(in_features, out_features, bias=bias is not None)

        quant_W, scale, row_min, orig_cols = quantize_4bit_per_row(weight)
        new_cls.quant_W = quant_W
        new_cls.scale = scale
        new_cls.row_min = row_min
        new_cls.orig_in_features = orig_cols

        if bias is not None:
            new_cls.bias = bias.astype(mx.float16)

        return new_cls

    @classmethod
    def from_packed(
        cls, quant_W, scale, row_min, orig_in_features: int, bias=None, dtype=mx.float16
    ) -> "QuantizedLinear":
        """
        Build directly from already-quantized weights
        Adapted to the way weights was saved -> mistral_qlora.model.convert_weights_mlx
        Given saved quantized weights:
            - quant_W, mx.array (out, in), dtype mx.uint8
            - scale, mx.array (out, ), dtype float
            - row_min, mx.array (out, ), dtype float
            - orig_in_features, int
            - dtype, mx.float16, insures dequant/quant never returns float32

        returns QuantizedLinear initialiazed with provided quantized weights.
        """
        quant_W = mx.array(quant_W, dtype=mx.uint8)
        scale = mx.array(scale, dtype=dtype)
        row_min = mx.array(row_min, dtype=dtype)

        out_features, _ = quant_W.shape
        in_features = int(orig_in_features)

        new_cls = cls(in_features, out_features, bias=bias is not None)

        new_cls.quant_W = quant_W
        new_cls.scale = scale
        new_cls.row_min = row_min
        new_cls.orig_in_features = in_features

        if bias is not None:
            new_cls.bias = mx.array(bias, dtype=dtype)

        return new_cls

    def _dequantize_weight(self) -> mx.array:
        """
        Dequantize the stored 4-bit weights to a float32 matrix.

        Returns:
            W_approx: float32 mlx.core.array
        """
        return dequantize_4bit_per_row(
            self.quant_W,
            self.scale,
            self.row_min,
            self.orig_in_features,
            dtype=mx.float16,
        )

    def __call__(self, x: mx.array, use_lora: bool = False) -> mx.array:
        """
        Performs linear forward pass.
        Dequantizes the frozen weights W, computes and returns Wx + (bias optional)

        Input: x mx.array
        use_lora: accepted and ignored; this layer holds no adapters.
        """
        W = self._dequantize_weight()
        y = x @ W.T

        if self.bias is not None:
            y = y + self.bias

        return y


class LoRALinear(nn.Module):
    """
    Wrap a QuantizedLinear layer (mlx-adapted 4-bit quantized layer) with LoRA adapters.


    y = (frozen_weights + adapter) * x

    - base:    QuantizedLinear
    - A:       (in_features x r)
    - B:       (r x out_features)
        --> Adding adapters to specific layers of the 7b model. Adapters consists of low rank (r)
        matrices A & B. During Training, only A & B are learned.
    """

    def __init__(
        self,
        base: QuantizedLinear,
        r: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()

        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.in_features = base.in_features
        self.out_features = base.out_features

        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.params_init()

    def params_init(self):
        """
        LoRA A & B matrices initialization:
                - A: small random normal distribution
                - B: zero
        """
        nn.init.normal(self.lora_A.weight, std=0.01)
        self.lora_B.weight = mx.zeros_like(self.lora_B.weight)

    @classmethod
    def from_quantLinear(
        cls,
        base: QuantizedLinear,
        r: int,
        alpha: float,
        dropout: float,
    ) -> "LoRALinear":
        """
        Build a LoRALinear on top of an existing QuantizedLinear.
        """
        return cls(base=base, r=r, alpha=alpha, dropout=dropout)

    def __call__(self, x: mx.array, use_lora: bool = True) -> mx.array:
        """
        Performs QLoRA forward pass on a Linear layer.
        """
        y = self.base(x)

        if use_lora and self.r > 0:
            if self.dropout is not None and self.training:
                x_lora = self.dropout(x)
            else:
                x_lora = x

            delta = self.lora_B(self.lora_A(x_lora))
            delta = delta * self.scaling

            y = y + delta

        return y
