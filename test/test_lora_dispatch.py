"""Resolution of `use_lora` and the shared projection signature."""

import mlx.core as mx
import pytest

from mistral_qlora.model.model_utils import (
    LORA_TARGETS,
    MistralAttention,
    MistralMLP,
    resolve_use_lora,
)
from mistral_qlora.quant.utils_linear import Linear, LoRALinear, QuantizedLinear


def test_bool_expands_to_every_projection():
    assert resolve_use_lora(True) == dict.fromkeys(LORA_TARGETS, True)
    assert resolve_use_lora(False) == dict.fromkeys(LORA_TARGETS, False)


def test_dict_passes_through_unchanged():
    selection = dict.fromkeys(LORA_TARGETS, False) | {"q": True}
    assert resolve_use_lora(selection) == selection


@pytest.mark.parametrize("layer_cls", [Linear, QuantizedLinear, LoRALinear])
def test_all_three_linears_accept_use_lora(layer_cls):
    """Linear, QuantizedLinear and LoRALinear share one call signature."""
    if layer_cls is Linear:
        layer = Linear(16, 8, bias=False)
    elif layer_cls is QuantizedLinear:
        layer = QuantizedLinear.convert_4bit(mx.random.normal((8, 16)))
    else:
        base = QuantizedLinear.convert_4bit(mx.random.normal((8, 16)))
        layer = LoRALinear.from_quantLinear(base, r=4, alpha=8, dropout=0.0)

    x = mx.random.normal((2, 16))
    for flag in (True, False):
        assert layer(x, use_lora=flag).shape == (2, 8)


def test_attention_accepts_its_own_default(tiny_config, packed_attn):
    """Attention is callable without an explicit `use_lora`."""
    attn = MistralAttention.from_quantized_weights(tiny_config, packed_attn)
    x = mx.random.normal((1, 4, tiny_config.hidden_size_atten))

    out, _ = attn(x)

    assert out.shape == x.shape


def test_mlp_accepts_its_own_default(tiny_config, packed_mlp):
    """The MLP is callable without an explicit `use_lora`."""
    mlp = MistralMLP.from_quantized_weights(tiny_config, packed_mlp)
    x = mx.random.normal((1, 4, tiny_config.hidden_size_atten))

    assert mlp(x).shape == x.shape


def test_bool_and_equivalent_dict_agree(tiny_config, packed_attn):
    attn = MistralAttention.from_quantized_weights(tiny_config, packed_attn)
    x = mx.random.normal((1, 4, tiny_config.hidden_size_atten))

    via_bool, _ = attn(x, use_lora=True)
    via_dict, _ = attn(x, use_lora=dict.fromkeys(LORA_TARGETS, True))

    assert mx.allclose(via_bool, via_dict).item()


def test_an_error_inside_an_adapter_propagates(tiny_config, packed_attn):
    """A failure in a LoRA forward pass raises rather than disabling the adapter."""
    attn = MistralAttention.from_quantized_weights(tiny_config, packed_attn)
    assert isinstance(attn.q_proj, LoRALinear)

    def exploding(_x):
        raise TypeError("shape error inside the adapter")

    attn.q_proj.lora_A = exploding

    with pytest.raises(TypeError, match="inside the adapter"):
        attn(mx.random.normal((1, 4, tiny_config.hidden_size_atten)), use_lora=True)
