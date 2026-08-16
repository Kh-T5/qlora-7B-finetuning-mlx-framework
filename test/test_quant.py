"""4-bit pack/unpack and the quantized linear layer."""

import mlx.core as mx
import pytest

from mistral_qlora.quant.quant_4bit import (
    dequantize_4bit_per_row,
    quantize_4bit_per_row,
)
from mistral_qlora.quant.utils_linear import LoRALinear, QuantizedLinear


def test_roundtrip_error_is_bounded_by_step_size():
    """Dequantized weights land within half a quantization step of the original.

    With 4 bits the step is (row_max - row_min) / 15, so the worst-case error from
    round-to-nearest is half that, with 1% slack for fp16 rounding.
    """
    w = mx.random.normal((16, 64)).astype(mx.float16)
    quant_w, scale, row_min, orig_in = quantize_4bit_per_row(w)
    w_hat = dequantize_4bit_per_row(quant_w, scale, row_min, orig_in)

    assert w_hat.shape == w.shape
    max_err = mx.max(mx.abs(w.astype(mx.float32) - w_hat.astype(mx.float32))).item()
    half_step = (mx.max(scale).item() / 2.0) * 1.01
    assert max_err <= half_step, f"{max_err} exceeds half-step {half_step}"


def test_packing_halves_the_column_count():
    """Two 4-bit values share each uint8, so 64 columns pack into 32."""
    w = mx.random.normal((8, 64))
    quant_w, _, _, orig_in = quantize_4bit_per_row(w)

    assert quant_w.dtype == mx.uint8
    assert quant_w.shape == (8, 32)
    assert orig_in == 64


def test_odd_column_count_is_padded_then_trimmed():
    """An odd width pads to ceil(7/2) bytes; the pad must not survive the roundtrip."""
    w = mx.random.normal((4, 7))
    quant_w, scale, row_min, orig_in = quantize_4bit_per_row(w)

    assert quant_w.shape == (4, 4)
    assert orig_in == 7
    assert dequantize_4bit_per_row(quant_w, scale, row_min, orig_in).shape == (4, 7)


@pytest.mark.xfail(
    strict=True,
    reason="B9: eps=1e-8 underflows to 0.0 in float16, so the scale floor is a "
    "no-op and near-constant rows quantize through a 0/0 division. Fixed in CS12.",
)
def test_constant_row_does_not_divide_by_zero():
    """A row with zero range would give scale = 0; the eps floor must catch it."""
    w = mx.zeros((3, 16))
    quant_w, scale, row_min, orig_in = quantize_4bit_per_row(w)
    w_hat = dequantize_4bit_per_row(quant_w, scale, row_min, orig_in)

    assert bool(mx.all(scale > 0).item())
    assert bool(mx.all(mx.isfinite(w_hat)).item())


@pytest.mark.parametrize("with_bias", [False, True])
def test_quantized_linear_matches_reference_matmul(with_bias):
    out_features, in_features = 32, 64
    w = mx.random.normal((out_features, in_features))
    bias = mx.random.normal((out_features,)) if with_bias else None

    layer = QuantizedLinear.convert_4bit(w, bias)
    x = mx.random.normal((3, in_features))

    got = layer(x)
    want = (
        x
        @ dequantize_4bit_per_row(
            layer.quant_W, layer.scale, layer.row_min, layer.orig_in_features
        )
        .astype(x.dtype)
        .T
    )
    if bias is not None:
        want = want + bias

    assert got.shape == (3, out_features)
    assert mx.allclose(got, want, atol=1e-2).item()


def test_from_packed_reconstructs_the_same_layer(pack):
    w = mx.random.normal((32, 64))
    packed = pack(w)

    layer = QuantizedLinear.from_packed(
        packed["weight_q"], packed["scale"], packed["row_min"], packed["orig_in"]
    )

    assert layer.in_features == 64
    assert layer.out_features == 32
    assert layer.quant_W.dtype == mx.uint8

    x = mx.random.normal((3, 64))
    assert bool(mx.all(mx.isfinite(layer(x))).item())


def test_lora_b_starts_at_zero_so_the_adapter_is_a_noop():
    """LoRA must not perturb the pretrained model before any training happens."""
    base = QuantizedLinear.convert_4bit(mx.random.normal((32, 64)))
    lora = LoRALinear.from_quantLinear(base, r=8, alpha=16, dropout=0.0)

    x = mx.random.normal((3, 64))
    assert bool(mx.all(lora.lora_B.weight == 0).item())
    assert mx.allclose(lora(x, use_lora=True), lora(x, use_lora=False)).item()


def test_lora_scaling_is_alpha_over_r():
    base = QuantizedLinear.convert_4bit(mx.random.normal((32, 64)))
    lora = LoRALinear.from_quantLinear(base, r=8, alpha=16, dropout=0.0)
    assert lora.scaling == 2.0
