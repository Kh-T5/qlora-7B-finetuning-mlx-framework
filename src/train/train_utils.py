import mlx.nn as nn
import mlx.core as mx
from src.model.model_wrapper import MistralForCausalLM
from src.quant.utils_linear import LoRALinear
from mlx.utils import tree_flatten


def make_lora_only_trainable(model: MistralForCausalLM):
    """
    Freeze everything, then unfreeze ONLY LoRA adapter params (lora_A, lora_B)
    inside each LoRALinear.
    """
    # 1) Freeze the whole tree
    model.freeze(recurse=True)

    # 2) For each LoRALinear, unfreeze its A/B submodules
    def unfreeze_lora(prefix, mod: nn.Module):
        if isinstance(mod, LoRALinear):
            # Make extra sure the base stays frozen
            if hasattr(mod, "base"):
                mod.base.freeze(recurse=True)

            # Unfreeze lora_A and lora_B fully (their weights)
            if hasattr(mod, "lora_A"):
                mod.lora_A.unfreeze(recurse=True)
            if hasattr(mod, "lora_B"):
                mod.lora_B.unfreeze(recurse=True)

    model.apply_to_modules(unfreeze_lora)
    n_total = sum(v.size for _, v in tree_flatten(model.parameters()))
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(
        f"Total params: {n_total/1e6:.3f}M, "
        f"trainable LoRA params: {n_train/1e6:.3f}M"
    )


def lm_loss_fn(model, batch, use_lora):
    """
    Returns loss during training over a batch.
    Inputs:
        - batch: dict with 'input_ids', 'attention_mask', 'labels'
        - model: MistralForCausalLM
        - use_lora: dict | bool
    """
    _, loss, _ = model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        caches=None,
        use_lora=use_lora,
    )
    return loss


def batch_token_loss_and_count(
    logits: mx.array, labels: mx.array, attention_mask: mx.array | None
):
    """
    Compute total CE loss over valid tokens and count them.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    if attention_mask is not None:
        shift_attn = attention_mask[:, 1:]
        valid_from_mask = shift_attn > 0
    else:
        valid_from_mask = mx.ones_like(shift_labels, dtype=mx.bool_)

    valid_from_labels = shift_labels != -100
    valid = valid_from_mask & valid_from_labels

    # Replace ignored labels by 0 to keep indices in range
    safe_labels = mx.where(
        valid,
        shift_labels,
        mx.zeros_like(shift_labels),
    )

    # Per-token CE, no reduction
    per_token = nn.losses.cross_entropy(
        shift_logits.astype(mx.float32),
        safe_labels.astype(mx.int32),
        axis=-1,
        reduction="none",
    )

    per_token = per_token * valid.astype(per_token.dtype)
    total_loss = mx.sum(per_token)
    total_tokens = mx.sum(valid.astype(mx.int32))

    return total_loss, total_tokens
