import mlx.nn as nn
from mlx.utils import tree_flatten

from mistral_qlora.model.model_wrapper import MistralForCausalLM
from mistral_qlora.quant.utils_linear import LoRALinear


def make_lora_only_trainable(model: MistralForCausalLM):
    """
    Freeze everything, then unfreeze ONLY LoRA adapter params (lora_A, lora_B)
    inside each LoRALinear.
    """
    model.freeze(recurse=True)

    def unfreeze_lora(prefix, mod: nn.Module):
        if isinstance(mod, LoRALinear):
            if hasattr(mod, "base"):
                mod.base.freeze(recurse=True)
            if hasattr(mod, "lora_A"):
                mod.lora_A.unfreeze(recurse=True)
                mod.lora_B.unfreeze(recurse=True)

    model.apply_to_modules(unfreeze_lora)

    n_total = sum(v.size for _, v in tree_flatten(model.parameters()))
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(
        f"Total params: {n_total / 1e6:.3f}M, "
        f"trainable LoRA params: {n_train / 1e6:.3f}M"
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
