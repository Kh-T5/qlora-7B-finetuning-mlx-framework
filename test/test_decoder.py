"""MLP, decoder layer, and the full decoder built from an on-disk checkpoint."""

import mlx.core as mx

from mistral_qlora.model.mistral_decoder import MistralDecoder, MistralDecoderLayer
from mistral_qlora.model.model_utils import MistralMLP


def test_mlp_preserves_the_input_width(tiny_config, packed_mlp, use_lora_off):
    mlp = MistralMLP.from_quantized_weights(tiny_config, packed_mlp)
    b, t, d = 2, 4, tiny_config.hidden_size_atten

    out = mlp(mx.random.normal((b, t, d)), use_lora=use_lora_off)

    assert out.shape == (b, t, d)


def test_decoder_layer_forward(
    tiny_config, packed_attn, packed_mlp, weights_norm, use_lora_off
):
    layer = MistralDecoderLayer.from_quantized_weights(
        tiny_config,
        packed_weights_mlp=packed_mlp,
        packed_weights_attn=packed_attn,
        weights_norm=weights_norm,
    )
    b, t, d = 2, 6, tiny_config.hidden_size_atten

    out, cache = layer(mx.random.normal((b, t, d)), use_lora=use_lora_off)

    assert out.shape == (b, t, d)
    assert cache["k"].shape[2] == t


def test_decoder_layer_is_residual(
    tiny_config, packed_attn, packed_mlp, weights_norm, use_lora_off
):
    """Zeroed output projections must make the layer an identity map.

    If the residual connections were dropped or misplaced, this returns zeros.
    """
    layer = MistralDecoderLayer.from_quantized_weights(
        tiny_config,
        packed_weights_mlp=packed_mlp,
        packed_weights_attn=packed_attn,
        weights_norm=weights_norm,
    )
    d, d_h = tiny_config.hidden_size_atten, tiny_config.hidden_size_mlp
    layer.attn.o_proj = lambda x, **_: mx.zeros_like(x)
    layer.mlp.down_proj = lambda x, **_: mx.zeros((*x.shape[:-1], d))

    x = mx.random.normal((1, 3, d))
    out, _ = layer(x, use_lora=use_lora_off)

    assert d_h != d, "stub must actually change width"
    assert mx.allclose(out, x, atol=1e-3).item()


def test_build_decoder_from_npz(tiny_config, checkpoint_dir, use_lora_off):
    """The on-disk checkpoint layout round-trips into a working decoder."""
    decoder = MistralDecoder.build_decoder_from_npz(
        tiny_config, checkpoint_dir["layers_dir"], checkpoint_dir["other_path"]
    )
    b, t, d = 2, 7, tiny_config.hidden_size_atten

    out, caches = decoder(mx.random.normal((b, t, d)), use_lora=use_lora_off)

    assert out.shape == (b, t, d)
    assert len(caches) == tiny_config.num_layers
    assert caches[0]["k"].shape[2] == t


def test_decoder_returns_one_cache_per_layer(tiny_config, checkpoint_dir, use_lora_off):
    decoder = MistralDecoder.build_decoder_from_npz(
        tiny_config, checkpoint_dir["layers_dir"], checkpoint_dir["other_path"]
    )
    x = mx.random.normal((1, 5, tiny_config.hidden_size_atten))

    _, caches = decoder(x, use_lora=use_lora_off)

    assert len(caches) == tiny_config.num_layers
    assert all("k" in c and "v" in c for c in caches)
