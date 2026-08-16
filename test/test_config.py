"""Configuration objects: validation, immutability and path layout."""

import dataclasses
from pathlib import Path

import pytest

from mistral_qlora.config import MistralConfig, Paths, TrainConfig
from mistral_qlora.constants import LORA_TARGETS


def test_defaults_describe_mistral_7b():
    cfg = MistralConfig()

    assert cfg.num_layers == 32
    assert cfg.hidden_size_atten == cfg.num_attention_heads * cfg.head_dim
    assert cfg.vocab_size == 32000


def test_head_dimensions_must_multiply_to_the_hidden_size():
    with pytest.raises(ValueError, match="hidden_size_atten"):
        MistralConfig(hidden_size_atten=64, num_attention_heads=5, head_dim=16)


def test_query_heads_must_be_a_multiple_of_kv_heads():
    with pytest.raises(ValueError, match="multiple of num_key_value_heads"):
        MistralConfig(
            hidden_size_atten=64,
            num_attention_heads=4,
            head_dim=16,
            num_key_value_heads=3,
        )


def test_lora_true_must_name_every_projection():
    with pytest.raises(ValueError, match="missing projections"):
        MistralConfig(lora_true={"q": True})


def test_configs_are_frozen():
    for cfg in (MistralConfig(), TrainConfig()):
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.__setattr__(dataclasses.fields(cfg)[0].name, 1)


def test_each_config_gets_its_own_lora_dict():
    a, b = MistralConfig(), MistralConfig()
    a.lora_true["q"] = False

    assert b.lora_true["q"] is True


def test_lora_targets_cover_attention_and_mlp():
    assert set(LORA_TARGETS) == {"q", "k", "v", "o", "gate", "up", "down"}


def test_paths_hang_off_a_single_root():
    paths = Paths(root=Path("/tmp/somewhere"))

    assert paths.quantized_layers.is_relative_to(paths.root)
    assert paths.quantized_other.is_relative_to(paths.root)
    assert paths.adapters_dir.is_relative_to(paths.root)
    assert paths.training_results.is_relative_to(paths.root)


def test_default_paths_are_under_data():
    assert Paths().root == Path("data")
