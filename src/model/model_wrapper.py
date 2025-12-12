import mlx.core as mx
import mlx.nn as nn
from src.model.model import MistralModel
from src.model.model_utils import MistralConfig


class MistralForCausalLM(nn.Module):
    """
    LM wrapper around MistralModel.

    - MistralModel: "backbone" that maps (input_ids, masks, caches) -> logits.
    - MistralForCausalLM: adds training logic (label shifting, loss).
    """

    def __init__(self, model: MistralModel):
        super().__init__()
        self.model = model

    @classmethod
    def from_mistral_7b(
        cls, config: MistralConfig, dir_weights_q: str, path_weights: str
    ) -> "MistralForCausalLM":
        """
        Constructor from saved, frozen weights.
        """
        base = MistralModel.from_mistral_7b(config, dir_weights_q, path_weights)
        return cls(base)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        labels: mx.array | None = None,
        caches=None,
        use_lora: dict | bool = False,
    ):
        """
        Inputs:
            - input_ids: Batch of token ids.

            - attention_mask: (B, T),
                1 for real tokens, 0 for padding.
                -> Used for masking loss and or building the attention mask inside the backbone.

            - labels: (B, T),
                Target token ids. If provided, we'll compute the LM loss.
                Convention: same shape as input_ids.
                We do the usual "predict token t from token t-1" shift.

            - caches:
                KV cache tree, one entry per layer (used during generation).
                Pass None during training.

            - use_lora: dict | bool
                Either:
                - a bool: enable/disable LoRA everywhere
                - a dict: fine-grained control
                    {"q": True, "k": True, "v": True, "o": False, ...}

        Returns:
            - logits: (B, T, vocab_size)
            - loss: scalar mx.array | None
            - new_caches: updated KV caches (for generation)
        """

        logits, new_caches = self.model(
            input_ids,
            attention_mask=attention_mask,
            caches=caches,
            use_lora=use_lora,
        )

        # Compute loss during training
        loss = None
        if labels is not None:

            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            if attention_mask is not None:
                shift_attn = attention_mask[:, 1:]  # (B, T-1)
                valid_from_mask = shift_attn > 0
            else:
                valid_from_mask = mx.ones_like(shift_labels, dtype=mx.bool_)

            valid_from_labels = shift_labels != -100
            valid = valid_from_mask & valid_from_labels  # (B, T-1) bool

            safe_labels = mx.where(
                valid,
                shift_labels,
                mx.zeros_like(shift_labels),
            )

            # Per-token CE loss
            per_token = nn.losses.cross_entropy(
                shift_logits,
                safe_labels.astype(mx.int32),
                axis=-1,
                reduction="none",
            )

            # Mask out ignored positions
            per_token = per_token * valid.astype(per_token.dtype)

            # Average over tokens
            denom = mx.maximum(mx.sum(valid), mx.array(1, dtype=mx.int32))
            loss = mx.sum(per_token) / denom

        return logits, loss, new_caches
