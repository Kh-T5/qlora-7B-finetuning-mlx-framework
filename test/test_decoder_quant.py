import os
import tempfile
import numpy as np
import mlx.core as mx

from src.quant.quant_4bit import quantize_4bit_per_row
from src.quant.utils_linear import QuantizedLinear
from src.model.model_utils import (
    MistralConfig,
    MistralAttention,
    MistralMLP,
)
from src.model.mistral_decoder import (
    MistralDecoderLayer,
    MistralDecoder,
)


def make_tiny_config() -> MistralConfig:
    """
    Build a "small" config that is cheaper to test with but
    obeys all architectural constraints:
        hidden_size_atten = num_heads * head_dim
    """
    cfg = MistralConfig()  # starts with real defaults

    cfg.hidden_size_atten = 64  # D
    cfg.hidden_size_mlp = 128  # intermediate size
    cfg.num_attention_heads = 4  # H
    cfg.num_key_value_heads = 2  # H_kv
    cfg.head_dim = 16  # Dh so that H * Dh = 64
    cfg.num_layers = 2  # tiny decoder

    return cfg


def pack_weight(W: mx.array):
    """
    Convenience helper for tests:
    full-precision -> 4-bit packed dict
    matching what from_packed expects.
    """
    qW, scale, row_min, orig_in = quantize_4bit_per_row(W)
    return {
        "weight_q": qW,
        "scale": scale,
        "row_min": row_min,
        "orig_in": int(orig_in),
    }


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------


def test_quantized_linear_from_packed():
    print("Running QuantizedLinear.from_packed test...")

    out_features = 32
    in_features = 64
    W = mx.random.normal((out_features, in_features))

    packed = pack_weight(W)

    qlin = QuantizedLinear.from_packed(
        quant_W=packed["weight_q"],
        scale=packed["scale"],
        row_min=packed["row_min"],
        orig_in_features=packed["orig_in"],
        bias=None,
    )

    x = mx.random.normal((3, in_features))
    y_hat = qlin(x)  # (3, 32)
    y_ref = x @ W.T  # full-precision reference

    assert y_hat.shape == y_ref.shape
    mae = mx.mean(mx.abs(y_ref - y_hat)).item()
    print(f"  MAE vs fp32: {mae:.4f}")
    assert np.isfinite(mae)

    print("  ✓ QuantizedLinear.from_packed OK\n")


def test_mistral_attention_from_packed():
    print("Running MistralAttention.from_quantized_weights test...")

    cfg = make_tiny_config()
    D = cfg.hidden_size_atten
    H = cfg.num_attention_heads
    H_kv = cfg.num_key_value_heads
    Dh = cfg.head_dim

    # Shapes match Linear(out_features, in_features)
    Wq = mx.random.normal((H * Dh, D))
    Wk = mx.random.normal((H_kv * Dh, D))
    Wv = mx.random.normal((H_kv * Dh, D))
    Wo = mx.random.normal((D, D))

    packed_attn = {
        "q_proj": pack_weight(Wq),
        "k_proj": pack_weight(Wk),
        "v_proj": pack_weight(Wv),
        "o_proj": pack_weight(Wo),
    }

    attn = MistralAttention.from_quantized_weights(cfg, packed_attn)

    B, T = 2, 5
    x = mx.random.normal((B, T, D))
    out, cache = attn(x)

    assert out.shape == (B, T, D)
    assert "k" in cache and "v" in cache
    assert cache["k"].shape == (B, H_kv, T, Dh)
    assert cache["v"].shape == (B, H_kv, T, Dh)

    print("  ✓ MistralAttention.from_quantized_weights OK\n")


def test_mistral_mlp_from_packed():
    print("Running MistralMLP.from_quantized_weights test...")

    cfg = make_tiny_config()
    D_in = cfg.hidden_size_atten
    D_h = cfg.hidden_size_mlp

    # Linear(out_features, in_features)
    W_gate = mx.random.normal((D_h, D_in))
    W_up = mx.random.normal((D_h, D_in))
    W_down = mx.random.normal((D_in, D_h))

    packed_mlp = {
        "gate_proj": pack_weight(W_gate),
        "up_proj": pack_weight(W_up),
        "down_proj": pack_weight(W_down),
    }

    mlp = MistralMLP.from_quantized_weights(cfg, packed_mlp)

    B, T = 2, 4
    x = mx.random.normal((B, T, D_in))
    out = mlp(x)

    assert out.shape == (B, T, D_in)
    print("  ✓ MistralMLP.from_quantized_weights OK\n")


def test_decoder_layer_from_packed():
    print("Running MistralDecoderLayer.from_quantized_weights test...")

    cfg = make_tiny_config()
    D = cfg.hidden_size_atten
    H = cfg.num_attention_heads
    H_kv = cfg.num_key_value_heads
    Dh = cfg.head_dim
    D_h = cfg.hidden_size_mlp

    # --- Attention weights (packed) ---
    Wq = mx.random.normal((H * Dh, D))
    Wk = mx.random.normal((H_kv * Dh, D))
    Wv = mx.random.normal((H_kv * Dh, D))
    Wo = mx.random.normal((D, D))
    packed_attn = {
        "q_proj": pack_weight(Wq),
        "k_proj": pack_weight(Wk),
        "v_proj": pack_weight(Wv),
        "o_proj": pack_weight(Wo),
    }

    # --- MLP weights (packed) ---
    W_gate = mx.random.normal((D_h, D))
    W_up = mx.random.normal((D_h, D))
    W_down = mx.random.normal((D, D_h))
    packed_mlp = {
        "gate_proj": pack_weight(W_gate),
        "up_proj": pack_weight(W_up),
        "down_proj": pack_weight(W_down),
    }

    # --- Norm weights (full precision) ---
    input_norm = mx.ones((D,), dtype=mx.float32)
    post_norm = mx.ones((D,), dtype=mx.float32)
    weights_norm = {
        "input": input_norm,
        "post_attention": post_norm,
    }

    layer = MistralDecoderLayer.from_quantized_weights(
        cfg,
        packed_weights_mlp=packed_mlp,
        packed_weights_attn=packed_attn,
        weights_norm=weights_norm,
    )

    B, T = 2, 6
    x = mx.random.normal((B, T, D))
    out, cache = layer(x)

    assert out.shape == (B, T, D)
    assert "k" in cache and "v" in cache
    assert cache["k"].shape[2] == T

    print("  ✓ MistralDecoderLayer.from_quantized_weights OK\n")


def test_full_decoder_with_npz_roundtrip():
    """
    End-to-end test:
    - create fake quantized weights
    - save them as .npz / .npy like your real pipeline
    - rebuild decoder with build_decoder_from_npz
    - run a forward pass
    """
    print("Running MistralDecoder.build_decoder_from_npz test...")

    cfg = make_tiny_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write per-layer files (quantized + norms)
        for i in range(cfg.num_layers):
            D = cfg.hidden_size_atten
            H = cfg.num_attention_heads
            H_kv = cfg.num_key_value_heads
            Dh = cfg.head_dim
            D_h = cfg.hidden_size_mlp

            # --- Attention ---
            Wq = mx.random.normal((H * Dh, D))
            Wk = mx.random.normal((H_kv * Dh, D))
            Wv = mx.random.normal((H_kv * Dh, D))
            Wo = mx.random.normal((D, D))

            for name, W in [
                ("q_proj", Wq),
                ("k_proj", Wk),
                ("v_proj", Wv),
                ("o_proj", Wo),
            ]:
                qW, scale, row_min, orig_in = quantize_4bit_per_row(W)
                path = os.path.join(tmpdir, f"layer_{i:02d}_{name}.npz")
                np.savez(
                    path,
                    weight_q=qW,
                    scale=scale,
                    row_min=row_min,
                    orig_in=orig_in,
                )

            # --- MLP ---
            W_gate = mx.random.normal((D_h, D))
            W_up = mx.random.normal((D_h, D))
            W_down = mx.random.normal((D, D_h))

            for name, W in [
                ("gate_proj", W_gate),
                ("up_proj", W_up),
                ("down_proj", W_down),
            ]:
                qW, scale, row_min, orig_in = quantize_4bit_per_row(W)
                path = os.path.join(tmpdir, f"layer_{i:02d}_{name}.npz")
                np.savez(
                    path,
                    weight_q=qW,
                    scale=scale,
                    row_min=row_min,
                    orig_in=orig_in,
                )

            # --- Norms ---
            input_norm = np.ones((D,), dtype=np.float32)
            post_norm = np.ones((D,), dtype=np.float32)

            np.save(
                os.path.join(tmpdir, f"layer_{i:02d}_input_layernorm.npy"),
                input_norm,
            )
            np.save(
                os.path.join(tmpdir, f"layer_{i:02d}_post_attention_layernorm.npy"),
                post_norm,
            )

        # Build decoder from that directory
        decoder = MistralDecoder.build_decoder_from_npz(cfg, tmpdir)

        B, T = 2, 7
        x = mx.random.normal((B, T, cfg.hidden_size_atten))
        out, caches = decoder(x)

        assert out.shape == (B, T, cfg.hidden_size_atten)
        assert len(caches) == cfg.num_layers
        assert caches[0]["k"].shape[2] == T

    print("  ✓ MistralDecoder.build_decoder_from_npz OK\n")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    test_quantized_linear_from_packed()
    test_mistral_attention_from_packed()
    test_mistral_mlp_from_packed()
    test_decoder_layer_from_packed()
    test_full_decoder_with_npz_roundtrip()

    print("All tests passed ✅")
