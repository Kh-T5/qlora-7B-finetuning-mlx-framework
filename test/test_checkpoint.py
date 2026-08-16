"""The on-disk checkpoint format: round-trips, metadata and refusal."""

import json
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from mistral_qlora.checkpoint import (
    FORMAT_VERSION,
    METADATA_KEY,
    CheckpointFormatError,
    layer_norm_path,
    layer_weight_path,
    load_adapters,
    load_embeddings,
    load_norm,
    load_quantized_weight,
    save_adapters,
    save_embeddings,
    save_norm,
    save_quantized_weight,
)
from mistral_qlora.quant.quant_4bit import quantize_4bit_per_row
from mistral_qlora.train.train_utils import make_lora_only_trainable


def test_filenames_are_zero_padded(tmp_path):
    assert layer_weight_path(tmp_path, 7, "q_proj").name == "layer_07_q_proj.npz"
    assert layer_norm_path(tmp_path, 31, "input").name == "layer_31_input_layernorm.npy"


def test_quantized_weight_round_trip(tmp_path):
    packed = quantize_4bit_per_row(mx.random.normal((8, 16)))
    path = layer_weight_path(tmp_path, 0, "q_proj")

    save_quantized_weight(path, *packed)
    loaded = load_quantized_weight(path)

    assert loaded["weight_q"].dtype == np.uint8
    assert loaded["orig_in"] == 16
    assert isinstance(loaded["orig_in"], int)
    assert np.array_equal(loaded["weight_q"], np.array(packed[0]))


def test_written_files_record_the_scheme(tmp_path):
    path = layer_weight_path(tmp_path, 0, "q_proj")
    save_quantized_weight(path, *quantize_4bit_per_row(mx.random.normal((4, 8))))

    with np.load(path) as data:
        meta = json.loads(str(data[METADATA_KEY]))

    assert meta["format_version"] == FORMAT_VERSION
    assert meta["scheme"] == "per_row"
    assert meta["bits"] == 4


def test_unversioned_files_are_refused(tmp_path):
    """Checkpoints written before versioning must fail loudly, not load as garbage."""
    path = tmp_path / "layer_00_q_proj.npz"
    np.savez(
        path,
        weight_q=np.zeros((4, 4), dtype=np.uint8),
        scale=np.ones(4),
        row_min=np.zeros(4),
        orig_in=np.int32(8),
    )

    with pytest.raises(CheckpointFormatError, match="predates checkpoint versioning"):
        load_quantized_weight(path)


def test_a_future_format_version_is_refused(tmp_path):
    path = tmp_path / "layer_00_q_proj.npz"
    np.savez(
        path,
        weight_q=np.zeros((4, 4), dtype=np.uint8),
        **{METADATA_KEY: np.array(json.dumps({"format_version": FORMAT_VERSION + 1}))},
    )

    with pytest.raises(CheckpointFormatError, match="format version"):
        load_quantized_weight(path)


def test_norm_round_trip(tmp_path):
    path = layer_norm_path(tmp_path, 3, "input")
    save_norm(path, np.ones(16, dtype=np.float32))

    loaded = load_norm(path)

    assert loaded.shape == (16,)
    assert loaded.dtype == mx.float16


def test_embeddings_round_trip(tmp_path):
    path = tmp_path / "norm_embed_head.npz"
    save_embeddings(
        path,
        norm=np.ones(8, dtype=np.float32),
        embed=np.zeros((16, 8), dtype=np.float32),
        head=np.ones((16, 8), dtype=np.float32),
    )

    loaded = load_embeddings(path)

    assert set(loaded) == {"norm", "embed", "head"}
    assert loaded["embed"].shape == (16, 8)


def test_adapters_round_trip_to_identical_logits(
    tmp_path, tiny_model, tiny_config, batch, use_lora_on
):
    make_lora_only_trainable(tiny_model)
    tiny_model.model.decoder.layers[0].attn.q_proj.lora_B.weight = mx.ones_like(
        tiny_model.model.decoder.layers[0].attn.q_proj.lora_B.weight
    )
    before, _, _ = tiny_model(batch["input_ids"], use_lora=use_lora_on)

    path = tmp_path / "adapters.npz"
    save_adapters(tiny_model, path, tiny_config)

    tiny_model.model.decoder.layers[0].attn.q_proj.lora_B.weight = mx.zeros_like(
        tiny_model.model.decoder.layers[0].attn.q_proj.lora_B.weight
    )
    load_adapters(tiny_model, path, tiny_config)
    after, _, _ = tiny_model(batch["input_ids"], use_lora=use_lora_on)

    assert mx.allclose(before, after).item()


def test_adapters_record_their_shape(tmp_path, tiny_model, tiny_config):
    make_lora_only_trainable(tiny_model)
    path = tmp_path / "adapters.npz"

    save_adapters(tiny_model, path, tiny_config)
    meta = load_adapters(tiny_model, path, tiny_config)

    assert meta["r"] == tiny_config.r
    assert meta["alpha"] == tiny_config.alpha
    assert meta["targets"] == ["k", "q", "v"]


def test_adapters_from_a_different_rank_are_refused(tmp_path, tiny_model, tiny_config):
    make_lora_only_trainable(tiny_model)
    path = tmp_path / "adapters.npz"
    save_adapters(tiny_model, path, tiny_config)

    with pytest.raises(CheckpointFormatError, match="expects r="):
        load_adapters(tiny_model, path, replace(tiny_config, r=tiny_config.r * 2))


def test_saving_a_model_with_no_adapters_raises(tmp_path, tiny_model):
    tiny_model.freeze(recurse=True)

    with pytest.raises(ValueError, match="no trainable parameters"):
        save_adapters(tiny_model, tmp_path / "adapters.npz", None)
