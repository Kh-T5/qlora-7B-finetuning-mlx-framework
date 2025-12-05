import math
import mlx.core as mx
import mlx.nn as nn
from src.model.model_utils import MistralAttention, MistralConfig


def build_config():
    return MistralConfig(
        alpha=16.0,
        dropout=0.0,
        r=16,
        hidden_size=4096,
        rms_norm_eps=1e-5,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        rope_theta=1e4,
    )


def test_basic_forward():
    print("Running basic forward test...")
    config = build_config()
    attn = MistralAttention(config)

    B, T = 2, 5
    x = mx.random.normal((B, T, config.hidden_size))

    out, cache = attn(x)

    assert out.shape == (B, T, config.hidden_size)
    assert "k" in cache and "v" in cache

    k, v = cache["k"], cache["v"]

    assert k.shape == (B, config.num_key_value_heads, T, config.head_dim)
    assert v.shape == (B, config.num_key_value_heads, T, config.head_dim)

    print("  ✓ basic forward ok")


def test_cache_growth():
    print("Running cache growth test...")
    config = build_config()
    attn = MistralAttention(config)

    B, T1, T2 = 1, 4, 3
    x1 = mx.random.normal((B, T1, config.hidden_size))
    x2 = mx.random.normal((B, T2, config.hidden_size))

    out1, cache1 = attn(x1)
    assert cache1["k"].shape[2] == T1

    out2, cache2 = attn(x2, cache=cache1)
    assert cache2["k"].shape[2] == T1 + T2
    assert cache2["v"].shape[2] == T1 + T2

    print("  ✓ cache growth ok")


def test_with_mask():
    print("Running attention mask test...")
    config = build_config()
    attn = MistralAttention(config)

    B, T = 2, 6
    x = mx.random.normal((B, T, config.hidden_size))

    # Example: mask future tokens (simple causal mask)
    # scores shape: (B, H, T, S), we build mask (1, 1, T, S)
    S = T
    mask = mx.full((1, 1, T, S), 0.0)
    for i in range(T):
        # Disallow attending to positions > i
        if i + 1 < T:
            mask[..., i, i + 1 :] = -1e9

    out, cache = attn(x, attn_mask=mask)
    assert out.shape == (B, T, config.hidden_size)
    print("  ✓ mask forward ok")


def test_from_quantized_weights():
    print("Running from_quantized_weights test...")
    config = build_config()

    # Build fake "full precision" weights with correct shapes
    D = config.hidden_size
    H = config.num_attention_heads
    H_kv = config.num_key_value_heads
    Dh = config.head_dim

    q_out = H * Dh
    kv_out = H_kv * Dh

    weights = {
        "q_proj": mx.random.normal((q_out, D)),
        "k_proj": mx.random.normal((kv_out, D)),
        "v_proj": mx.random.normal((kv_out, D)),
        "o_proj": mx.random.normal((D, D)),
    }

    attn_q = MistralAttention.from_quantized_weights(config, weights)

    B, T = 2, 5
    x = mx.random.normal((B, T, D))

    out, cache = attn_q(x)
    assert out.shape == (B, T, D)
    assert cache["k"].shape == (B, H_kv, T, Dh)

    print("✓ from_quantized_weights ok")


if __name__ == "__main__":
    test_basic_forward()
    test_cache_growth()
    test_with_mask()
    test_from_quantized_weights()
    print("All tests passed")
