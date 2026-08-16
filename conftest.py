"""Shared pytest fixtures.

Everything here builds a *tiny* Mistral (2 layers, 64 dims, 128-token vocab) so the
whole suite runs in seconds with no Mistral-7B weights and no Hugging Face download.

`use_lora` accepts both a bool and a per-projection dict. The fixtures here supply
dicts for explicit control.
"""

import mlx.core as mx
import numpy as np
import pytest

from mistral_qlora.model.model_utils import MistralConfig
from mistral_qlora.model.model_wrapper import MistralForCausalLM
from mistral_qlora.quant.quant_4bit import quantize_4bit_per_row

ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ALL_LORA_KEYS = ("q", "k", "v", "o", "gate", "up", "down")


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed MLX before every test so results do not depend on execution order."""
    mx.random.seed(0)


@pytest.fixture
def use_lora_off() -> dict:
    """Every projection with adapters disabled."""
    return dict.fromkeys(ALL_LORA_KEYS, False)


@pytest.fixture
def use_lora_on() -> dict:
    """Every projection with adapters enabled."""
    return dict.fromkeys(ALL_LORA_KEYS, True)


@pytest.fixture
def tiny_config() -> MistralConfig:
    """A Mistral small enough to be free, still obeying the architecture's constraints.

    hidden_size_atten (D) must equal num_attention_heads (H) * head_dim (Dh), and
    embed_dim must equal it too for the decoder to accept the embedding output.
    num_key_value_heads < num_attention_heads so GQA expansion is exercised.
    Dropout is zero to keep every test deterministic.
    """
    cfg = MistralConfig()
    cfg.vocab_size = 128
    cfg.embed_dim = 64
    cfg.hidden_size_atten = 64
    cfg.hidden_size_mlp = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 16
    cfg.num_layers = 2
    cfg.dropout = 0.0
    return cfg


def pack_weight(w: mx.array) -> dict:
    """Full-precision weight -> the packed dict `from_packed` expects."""
    quant_w, scale, row_min, orig_in = quantize_4bit_per_row(w)
    return {
        "weight_q": quant_w,
        "scale": scale,
        "row_min": row_min,
        "orig_in": int(orig_in),
    }


@pytest.fixture
def pack():
    return pack_weight


def _attn_shapes(cfg: MistralConfig) -> dict:
    d, h, h_kv, dh = (
        cfg.hidden_size_atten,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    )
    return {
        "q_proj": (h * dh, d),
        "k_proj": (h_kv * dh, d),
        "v_proj": (h_kv * dh, d),
        "o_proj": (d, d),
    }


def _mlp_shapes(cfg: MistralConfig) -> dict:
    d, d_h = cfg.hidden_size_atten, cfg.hidden_size_mlp
    return {
        "gate_proj": (d_h, d),
        "up_proj": (d_h, d),
        "down_proj": (d, d_h),
    }


@pytest.fixture
def packed_attn(tiny_config) -> dict:
    return {
        name: pack_weight(mx.random.normal(shape))
        for name, shape in _attn_shapes(tiny_config).items()
    }


@pytest.fixture
def packed_mlp(tiny_config) -> dict:
    return {
        name: pack_weight(mx.random.normal(shape))
        for name, shape in _mlp_shapes(tiny_config).items()
    }


@pytest.fixture
def weights_norm(tiny_config) -> dict:
    ones = mx.ones((tiny_config.hidden_size_atten,), dtype=mx.float32)
    return {"input": ones, "post_attention": ones}


@pytest.fixture
def checkpoint_dir(tmp_path, tiny_config, monkeypatch):
    """Write a tiny checkpoint in the on-disk layout and return its paths.

    `build_decoder_from_npz` reads `mistral_other_layers_quant_path` from its module
    rather than from its `dir` argument, so that name is patched to keep the fixture
    from reaching the real checkpoint under data/.
    """
    cfg = tiny_config
    layers_dir = tmp_path / "decoder_mlp_layers"
    layers_dir.mkdir()

    shapes = {**_attn_shapes(cfg), **_mlp_shapes(cfg)}
    for i in range(cfg.num_layers):
        for name, shape in shapes.items():
            quant_w, scale, row_min, orig_in = quantize_4bit_per_row(
                mx.random.normal(shape)
            )
            np.savez(
                layers_dir / f"layer_{i:02d}_{name}.npz",
                weight_q=quant_w,
                scale=scale,
                row_min=row_min,
                orig_in=orig_in,
            )
        norm = np.ones((cfg.hidden_size_atten,), dtype=np.float32)
        np.save(layers_dir / f"layer_{i:02d}_input_layernorm.npy", norm)
        np.save(layers_dir / f"layer_{i:02d}_post_attention_layernorm.npy", norm)

    other_path = tmp_path / "norm_embed_head.npz"
    np.savez(
        other_path,
        norm_np=np.ones((cfg.hidden_size_atten,), dtype=np.float32),
        embed_np=np.random.randn(cfg.vocab_size, cfg.embed_dim).astype(np.float32),
        head_np=np.random.randn(cfg.vocab_size, cfg.embed_dim).astype(np.float32),
    )

    monkeypatch.setattr(
        "mistral_qlora.model.mistral_decoder.mistral_other_layers_quant_path",
        str(other_path),
    )
    return {"layers_dir": str(layers_dir), "other_path": str(other_path)}


@pytest.fixture
def tiny_model(tiny_config, checkpoint_dir) -> MistralForCausalLM:
    """A full MistralForCausalLM built through the real from_mistral_7b path."""
    return MistralForCausalLM.from_mistral_7b(
        tiny_config,
        checkpoint_dir["layers_dir"],
        checkpoint_dir["other_path"],
    )


@pytest.fixture
def batch(tiny_config):
    """A supervised batch of 2 x 8 tokens.

    The first three label positions are -100, standing in for prompt tokens that
    must not contribute to the loss.
    """
    b, t = 2, 8
    input_ids = mx.random.randint(0, tiny_config.vocab_size, (b, t))
    labels = mx.array(np.array(input_ids))
    labels = mx.concatenate(
        [mx.full((b, 3), -100, dtype=labels.dtype), labels[:, 3:]], axis=1
    )
    attention_mask = mx.ones((b, t), dtype=mx.int32)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
