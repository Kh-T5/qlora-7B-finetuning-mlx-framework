"""Which parameters receive gradients.

QLoRA is correct only if the 4-bit base stays frozen and gradient reaches the
adapters alone.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mistral_qlora.train.train_utils import lm_loss_fn, make_lora_only_trainable


def _grads(model, batch, use_lora):
    """Return (loss, grad tree, flattened name -> grad).

    The tree is what `optimizer.update` needs; the flat dict is what assertions
    read. Passing the flat dict to `update` raises a KeyError deep inside
    tree_unflatten, which is a confusing way to find that out.
    """
    loss_and_grad = nn.value_and_grad(model, lm_loss_fn)
    loss, grads = loss_and_grad(model, batch, use_lora)
    mx.eval(loss, grads)
    return loss, grads, dict(tree_flatten(grads))


def test_only_lora_parameters_are_trainable(tiny_model):
    make_lora_only_trainable(tiny_model)

    names = [name for name, _ in tree_flatten(tiny_model.trainable_parameters())]

    assert names, "no trainable parameters at all"
    offenders = [n for n in names if "lora_A" not in n and "lora_B" not in n]
    assert not offenders, f"non-LoRA parameters are trainable: {offenders[:5]}"


def test_quantized_base_weights_are_never_trainable(tiny_model):
    make_lora_only_trainable(tiny_model)

    names = [name for name, _ in tree_flatten(tiny_model.trainable_parameters())]

    for forbidden in ("quant_W", "scale", "row_min", "embed", "lm_head"):
        assert not any(forbidden in n for n in names), f"{forbidden} must stay frozen"


def test_gradients_reach_only_the_adapters(tiny_model, batch, use_lora_on):
    """Gradient reaches the adapters and nothing else."""
    make_lora_only_trainable(tiny_model)
    _, _, grads = _grads(tiny_model, batch, use_lora_on)

    assert grads, "value_and_grad returned no gradients"
    stray = [n for n in grads if "lora_A" not in n and "lora_B" not in n]
    assert not stray, f"gradient leaked to non-LoRA parameters: {stray[:5]}"

    b_grads = {n: g for n, g in grads.items() if "lora_B" in n}
    assert b_grads, "no lora_B gradients at all"

    dead = [n for n, g in b_grads.items() if mx.max(mx.abs(g)).item() == 0.0]
    assert not dead, f"adapters received zero gradient: {dead[:5]}"


def test_every_targeted_projection_receives_gradient(
    tiny_model, tiny_config, batch, use_lora_on
):
    """Every adapted projection in every layer receives gradient.

    Asserted per projection: a whole-model check passes while individual
    adapters are dead.
    """
    make_lora_only_trainable(tiny_model)
    _, _, grads = _grads(tiny_model, batch, use_lora_on)

    targeted = [k for k, on in tiny_config.lora_true.items() if on]
    assert targeted, "fixture config adapts no projections"

    for layer_idx in range(tiny_config.num_layers):
        for proj in targeted:
            key = f"layers.{layer_idx}.attn.{proj}_proj.lora_B.weight"
            matches = [n for n in grads if n.endswith(key)]
            assert matches, f"no gradient for {proj}_proj in layer {layer_idx}"
            assert mx.max(mx.abs(grads[matches[0]])).item() > 0.0


def test_a_training_step_changes_only_adapter_weights(tiny_model, batch, use_lora_on):
    """End-to-end: one optimizer step must move adapters and nothing else."""
    import mlx.optimizers as optim

    make_lora_only_trainable(tiny_model)
    before = dict(tree_flatten(tiny_model.parameters()))
    frozen_before = {n: mx.array(p) for n, p in before.items() if "lora_" not in n}

    opt = optim.AdamW(learning_rate=1e-3)
    _, grad_tree, _ = _grads(tiny_model, batch, use_lora_on)
    opt.update(tiny_model, grad_tree)
    mx.eval(tiny_model.parameters(), opt.state)

    after = dict(tree_flatten(tiny_model.parameters()))
    for name, original in frozen_before.items():
        assert mx.array_equal(original, after[name]).item(), f"{name} moved"

    moved = [
        n
        for n in after
        if "lora_B" in n and not mx.array_equal(before[n], after[n]).item()
    ]
    assert moved, "no adapter weights changed after an optimizer step"
