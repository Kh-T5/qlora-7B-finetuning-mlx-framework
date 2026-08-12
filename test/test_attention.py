"""MistralAttention: shapes, GQA expansion, KV cache and masking."""

import mlx.core as mx

from mistral_qlora.model.model_utils import MistralAttention


def test_forward_shapes_and_cache(tiny_config, packed_attn, use_lora_off):
    attn = MistralAttention.from_quantized_weights(tiny_config, packed_attn)
    b, t, d = 2, 5, tiny_config.hidden_size_atten

    out, cache = attn(mx.random.normal((b, t, d)), use_lora=use_lora_off)

    assert out.shape == (b, t, d)
    # Cache holds pre-expansion KV heads, not query heads.
    expected = (b, tiny_config.num_key_value_heads, t, tiny_config.head_dim)
    assert cache["k"].shape == expected
    assert cache["v"].shape == expected


def test_cache_grows_by_the_new_sequence_length(tiny_config, packed_attn, use_lora_off):
    attn = MistralAttention.from_quantized_weights(tiny_config, packed_attn)
    d = tiny_config.hidden_size_atten
    t1, t2 = 4, 3

    _, cache1 = attn(mx.random.normal((1, t1, d)), use_lora=use_lora_off)
    assert cache1["k"].shape[2] == t1

    _, cache2 = attn(mx.random.normal((1, t2, d)), cache=cache1, use_lora=use_lora_off)
    assert cache2["k"].shape[2] == t1 + t2
    assert cache2["v"].shape[2] == t1 + t2


def test_cached_step_matches_the_full_forward(tiny_config, packed_attn, use_lora_off):
    """Decoding token-by-token with a cache must equal one full-sequence pass.

    This is the only check that exercises the RoPE position offset and the cache
    concatenation together — training never uses the cache, so nothing else would
    catch a drift here.
    """
    attn = MistralAttention.from_quantized_weights(tiny_config, packed_attn)
    d = tiny_config.hidden_size_atten
    t = 4
    x = mx.random.normal((1, t, d))

    # The full pass needs an explicit causal mask to be comparable: incremental
    # decoding is causal by construction, but a full forward without a mask is
    # bidirectional and would legitimately differ.
    causal = mx.triu(mx.full((t, t), -1e9), k=1)[None, None]
    full, _ = attn(x, attn_mask=causal, use_lora=use_lora_off)

    cache = None
    steps = []
    for i in range(t):
        step, cache = attn(x[:, i : i + 1, :], cache=cache, use_lora=use_lora_off)
        steps.append(step)
    incremental = mx.concatenate(steps, axis=1)

    assert incremental.shape == full.shape
    assert mx.allclose(incremental, full, atol=2e-2).item()


def test_masked_positions_are_ignored(tiny_config, packed_attn, use_lora_off):
    """Masking a position to -inf must make the output independent of its content."""
    attn = MistralAttention.from_quantized_weights(tiny_config, packed_attn)
    b, t, d = 1, 4, tiny_config.hidden_size_atten
    x = mx.random.normal((b, t, d))

    # Allow everything except the final key position.
    mask = mx.zeros((1, 1, t, t))
    mask[..., -1] = -1e9

    out_a, _ = attn(x, attn_mask=mask, use_lora=use_lora_off)

    x_perturbed = mx.array(x)
    x_perturbed[:, -1, :] = mx.random.normal((b, d)) * 100.0
    out_b, _ = attn(x_perturbed, attn_mask=mask, use_lora=use_lora_off)

    # Every position except the last (which is itself perturbed) is unchanged.
    assert mx.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-2).item()


def test_gqa_expands_kv_heads_to_query_heads(tiny_config, packed_attn, use_lora_off):
    assert tiny_config.num_attention_heads > tiny_config.num_key_value_heads

    attn = MistralAttention.from_quantized_weights(tiny_config, packed_attn)
    b, t, d = 2, 6, tiny_config.hidden_size_atten
    out, cache = attn(mx.random.normal((b, t, d)), use_lora=use_lora_off)

    assert out.shape == (b, t, d)
    assert cache["k"].shape[1] == tiny_config.num_key_value_heads


def test_lora_changes_the_output_once_b_is_nonzero(
    tiny_config, packed_attn, use_lora_on, use_lora_off
):
    attn = MistralAttention.from_quantized_weights(tiny_config, packed_attn)
    x = mx.random.normal((1, 4, tiny_config.hidden_size_atten))

    # lora_B is zero-initialised, so the adapter is a no-op until it is trained.
    off, _ = attn(x, use_lora=use_lora_off)
    on, _ = attn(x, use_lora=use_lora_on)
    assert mx.allclose(off, on).item()

    attn.q_proj.lora_B.weight = mx.random.normal(attn.q_proj.lora_B.weight.shape) * 0.1
    on_trained, _ = attn(x, use_lora=use_lora_on)
    assert not mx.allclose(off, on_trained, atol=1e-4).item()
